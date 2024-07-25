import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F
from torch.nn import init
from torch_scatter import scatter_mean, scatter_max
import numpy as np
import math


import mcubes
def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def get_cube_verts (res=32 ,device = 'cuda'):
    cube_verts = 1.1 * make_3d_grid(
        (-0.5,) * 3, (0.5,) * 3, (res,) * 3
    )
    return cube_verts.to(device)
# ---------------common ---------
def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6)  # (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    # if there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    p_nor = p / (1 + padding + 10e-4)  # (-0.5, 0.5)
    p_nor = p_nor + 0.5  # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

class positional_encoding(object):
    ''' Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    '''
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function

        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # chagne to the range [-1, 1]
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p

class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s # always possitive
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!
        p = self.pe(p)
        return p

def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


# --------------------- layers ---------------------

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# --------------------- unet ------------------------
def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

# --------------- encoder-----------------
class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max',
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        # if unet3d:
        #     self.unet3d = UNet3D(**unet3d_kwargs)
        # else:
        #     self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid,
                                    self.reso_grid)  # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')


        net = self.fc_pos(p)
        # print('self.fc_pos',self.fc_pos)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

# --------------- decoder-----------------

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,out_dim = 1,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(
            -1).squeeze(-1)
        return c

    def forward(self, p, c_plane, condition = None,**kwargs):
        '''
        :param p:
        :param c_plane:
        :param condition: added by Qiao
        :param kwargs:
        :return:
        '''
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        # print('condition',condition)
        if condition is not None:
            p = torch.cat((p,condition),2).float() # B,N,3 + B,N,C = B,N,3+C
            # print('p.shape',p.shape)
            # print('self.fc_p',self.fc_p)
        else:
            p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


decoder_dict = {
    'simple_local': LocalDecoder, ## shapenet_3plane.yaml
    # 'simple_local_crop': PatchLocalDecoder,
    # 'simple_local_point': LocalPointDecoder
}

encoder_dict = {
    'pointnet_local_pool': LocalPoolPointnet, ### shapenet_3plane.yaml
    # 'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    # 'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    # 'voxel_simple_local': voxels.LocalVoxelEncoder,
}


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder= None, encoder=None, device=None):
        super().__init__()
        if decoder is None:
            dim = 3  # cfg['data']['dim']
            c_dim = 32  # cfg['model']['c_dim']
            padding = 0.1  # cfg['data']['padding']
            decoder_kwargs = \
                {'sample_mode': 'bilinear', 'hidden_size': 32, 'out_dim': 1}  # modify here: add out_dim =3
            encoder_kwargs = \
                {'hidden_dim': 32, 'plane_type': ['xz', 'xy', 'yz'], 'plane_resolution': 64, 'unet': True,
                 'unet_kwargs': {'depth': 4, 'merge_mode': 'concat', 'start_filts': 32}}

            encoder = LocalPoolPointnet(
                dim=3, c_dim=c_dim, padding=padding,  # modify here: dim =6
                **encoder_kwargs)

            decoder = LocalDecoder(
                dim=dim, c_dim=c_dim, padding=padding,
                **decoder_kwargs
            )


        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        # p_r = self.decode(p, c, **kwargs)
        p_r = self.decode(p, c, **kwargs).logits
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        # p_r = dist.Bernoulli(logits=logits)
        # return p_r
        return logits

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


def regularize_geo(deformation,sdf):
    # Step 2: Normalize the deformation to avoid the flipped triangles.
    deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
    sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

    ####
    # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
    pos_shape = torch.sum((sdf.squeeze(dim=-1) > 0).int(), dim=-1)
    neg_shape = torch.sum((sdf.squeeze(dim=-1) < 0).int(), dim=-1)
    zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
    if torch.sum(zero_surface).item() > 0:
        update_sdf = torch.zeros_like(sdf[0:1])
        max_sdf = sdf.max()
        min_sdf = sdf.min()
        update_sdf[:, self.dmtet_geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
        update_sdf[:, self.dmtet_geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
        new_sdf = torch.zeros_like(sdf)
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                new_sdf[i_batch:i_batch + 1] += update_sdf
        update_mask = (new_sdf == 0).float()
        # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
        sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
        sdf_reg_loss = sdf_reg_loss * zero_surface.float()
        sdf = sdf * update_mask + new_sdf * (1 - update_mask)

    # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
    final_sdf = []
    final_def = []
    for i_batch in range(zero_surface.shape[0]):
        if zero_surface[i_batch]:
            final_sdf.append(sdf[i_batch: i_batch + 1].detach())
            final_def.append(deformation[i_batch: i_batch + 1].detach())
        else:
            final_sdf.append(sdf[i_batch: i_batch + 1])
            final_def.append(deformation[i_batch: i_batch + 1])
    sdf = torch.cat(final_sdf, dim=0)
    deformation = torch.cat(final_def, dim=0)
    return sdf, deformation, sdf_reg_loss

if __name__ == '__main__':
    def quick_test():
        device = torch.device('cuda', 1)
        '''random data'''
        B = 2
        N = 2048
        p_coords = torch.randn(B, N, 3).to(device)

        occ_points = torch.randn(B, N, 3).to(device)
        occ = torch.randn(B, N).to(device)

        '''construct network'''
        decoder = 'simple_local'  # cfg['model']['decoder'] # LocalDecoder
        encoder = 'pointnet_local_pool'  # cfg['model']['encoder'] # LocalPoolPointnet
        dim = 3  # cfg['data']['dim']
        c_dim = 32  # cfg['model']['c_dim']
        padding = 0.1  # cfg['data']['padding']
        decoder_kwargs = \
            {'sample_mode': 'bilinear', 'hidden_size': 32}
        encoder_kwargs = \
            {'hidden_dim': 32, 'plane_type': ['xz', 'xy', 'yz'], 'plane_resolution': 64, 'unet': True,
             'unet_kwargs': {'depth': 4, 'merge_mode': 'concat', 'start_filts': 32}}

        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs)

        decoder = decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **decoder_kwargs
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        ).to(device)


        '''forward'''
        inputs = p_coords
        print('inputs', inputs.shape)
        c = model.encode_inputs(inputs)  #
        for k, v in c.items():
            print('c', k, v.shape)

        # xz torch.Size([2, 32, 64, 64])
        # xy torch.Size([2, 32, 64, 64])
        # yz torch.Size([2, 32, 64, 64])

        kwargs = {}
        # General points
        logits = model.decode(occ_points, c, **kwargs) #.logits  # B,N
        print('logits.shape', logits.shape)

        '''check metrics'''
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        # loss = loss_i.sum(-1).mean()
        loss = loss_i.mean()

        accu = (occ.long() == (logits > 0).long()).float().mean()
        print('loss', loss)
        print('accu', accu)

    def test_pretrained():
        device = torch.device('cuda', 1)
        B = 1
        N = 2048
        '''construct network'''
        decoder = 'simple_local'  # cfg['model']['decoder'] # LocalDecoder
        encoder = 'pointnet_local_pool'  # cfg['model']['encoder'] # LocalPoolPointnet
        dim = 3  # cfg['data']['dim']
        c_dim = 32  # cfg['model']['c_dim']
        padding = 0.1  # cfg['data']['padding']
        decoder_kwargs = \
            {'sample_mode': 'bilinear', 'hidden_size': 32}
        encoder_kwargs = \
            {'hidden_dim': 32, 'plane_type': ['xz', 'xy', 'yz'], 'plane_resolution': 64, 'unet': True,
             'unet_kwargs': {'depth': 4, 'merge_mode': 'concat', 'start_filts': 32}}

        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs)

        decoder = decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **decoder_kwargs
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        ).to(device)

        '''load pretrained weights'''
        pretrain_dict = torch.load('datasets/pretrained/convonet/shapenet_3plane.pt')['model']
        model.load_state_dict(pretrain_dict, strict=True)
        print(f"Loaded pretrained convonet: #params = {sum([p.numel() for p in model.parameters()])}")
        # freeze
        # for param in self.generator.parameters():
        #     param.requires_grad = False
        # self.generator.eval()

        '''load real data'''
        from training.dataset import Dataset
        from torch.utils.data import DataLoader

        test_set = Dataset(split='test', point_per_shape=N)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=B,
                                 shuffle=False)
        i = 0
        for data in (test_loader):
            print(i, '/', len(test_loader))
            # fetch data
            occ_points, occ, voxels, more_images, \
            p_coords, p_colors, p_material_ids, p_uvs, \
            mat_images, mat_colors, \
            mat_means_rgb, mat_vars_rgb, mat_means_xyz, mat_vars_xyz, mat_area_ratios, \
            mat_images_ids, mat_colors_ids, p_img_mat_arr_id, p_color_mat_arr_id, mat_cls, names = data
            occ_points = occ_points.to(device)
            occ = occ.to(device)
            p_coords = p_coords.to(device)
            p_colors = p_colors.to(device)

            # print(p_coords.shape)


            '''forward'''
            inputs = p_coords
            print('inputs', inputs.shape)
            c = model.encode_inputs(inputs)  #
            for k, v in c.items():
                print('c', k, v.shape)

            # xz torch.Size([2, 32, 64, 64])
            # xy torch.Size([2, 32, 64, 64])
            # yz torch.Size([2, 32, 64, 64])

            kwargs = {}
            # General points
            logits = model.decode(occ_points, c, **kwargs)#.logits  # B,N
            print('logits.shape', logits.shape)

            '''check metrics'''
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
            # loss = loss_i.sum(-1).mean()
            loss = loss_i.mean()

            accu = (occ.long() == (logits > 0).long()).float().mean()
            print('loss', loss)
            print('accu', accu)

            '''vis'''
            import trimesh
            use_DMTet = False
            if use_DMTet:
                from models.networks_get3d import DMTetGeometry

                dmtet_geometry = DMTetGeometry(grid_res=90, scale=1.0, device=device)  # res = 64,70,80,90,100

                positions = dmtet_geometry.verts.unsqueeze(dim=0).expand(B, -1, -1).to(device)
                occ_logits = sdf = logits = model.decode(positions, c, **kwargs)#.logits  # B,N

                v_deformed = positions
                tets = dmtet_geometry.indices

                v_list = []
                f_list = []
                mesh_list = []
                # Using marching tet to obtain the mesh
                for i_batch in range(B):
                    verts, faces = dmtet_geometry.get_mesh(
                        v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                        with_uv=False, indices=tets)
                    ################################################################
                    from utils.other_utils import rotate_pc, normalize_pc

                    verts = normalize_pc(verts)
                    # verts = rotate_pc(verts,theta=-0.5*np.pi,axis='y') # N,3
                    # print('verts',verts.max().item(),verts.min().item()) # 0.444, -0.456 ...

                    ################################################################
                    v_list.append(verts)
                    f_list.append(faces)

                    mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                    vis = True
                    if vis:
                        print('i_batch', i_batch)
                        from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk

                        in_points = positions[i_batch][sdf[i_batch] > 0]
                        # in_points = normalize_pc(in_points)
                        # in_points = rotate_pc(in_points, theta=-0.5 * np.pi, axis='y')  # N,3
                        vis_actors_vtk([
                            get_mesh_actor_vtk(mesh, opacity=0.5),
                            # get_colorful_pc_actor_vtk(in_points, opacity=0.9)
                        ], arrows=True)
                    mesh_list.append(mesh)
            else:
                res = 128
                positions = get_cube_verts(res, device)
                positions = positions.unsqueeze(0).repeat(B, 1, 1)
                occ_logits = sdf = model.decode(positions, c, **kwargs)  # .logits  # B,N
                print('sdf.shape',sdf.shape)
                for i_batch in range(B):
                    sdf_i = sdf[i_batch].reshape(res,res,res)
                    vertices, triangles = mcubes.marching_cubes(
                        sdf_i.detach().cpu().numpy(), 0)

                    center = (vertices.max(0) + vertices.min(0)) / 2
                    max_l = (vertices.max(0) - vertices.min(0)).max()
                    vertices = ((vertices - center) / max_l)  # * 0.9

                    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                    vis = True
                    if vis:
                        print('i_batch', i_batch)
                        from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk

                        # in_points = positions[i_batch][sdf[i_batch] > 0]
                        # in_points = normalize_pc(in_points)
                        # in_points = rotate_pc(in_points, theta=-0.5 * np.pi, axis='y')  # N,3
                        vis_actors_vtk([
                            get_mesh_actor_vtk(mesh, opacity=1.0),
                            # get_colorful_pc_actor_vtk(in_points, opacity=0.9)
                        ], arrows=True)
                    # mesh_list.append(mesh)

    def test_modified():
        device = torch.device('cuda', 1)
        '''random data'''
        B = 2
        N = 2048
        p_coords = torch.randn(B, N, 3).to(device)
        p_colors = torch.randn(B, N, 3).to(device)


        '''construct network'''
        model = ConvolutionalOccupancyNetwork(
            device=device
        ).to(device)

        '''forward'''
        inputs = torch.cat((p_coords,p_colors),2)
        print('inputs', inputs.shape)
        c = model.encode_inputs(inputs)  #
        for k, v in c.items():
            print('c', k, v.shape)

        # xz torch.Size([2, 32, 64, 64])
        # xy torch.Size([2, 32, 64, 64])
        # yz torch.Size([2, 32, 64, 64])

        kwargs = {}
        # General points
        logits = model.decode(p_coords, c, **kwargs)#.logits  # B,N
        print('logits.shape', logits.shape)

        # '''check metrics'''
        # loss_i = F.binary_cross_entropy_with_logits(
        #     logits, occ, reduction='none')
        # # loss = loss_i.sum(-1).mean()
        # loss = loss_i.mean()
        #
        # accu = (occ.long() == (logits > 0).long()).float().mean()
        # print('loss', loss)
        # print('accu', accu)

    def test_dmtet():
        device = torch.device('cuda', 1)
        B = 2
        N = 2048
        '''construct network'''
        decoder = 'simple_local'  # cfg['model']['decoder'] # LocalDecoder
        encoder = 'pointnet_local_pool'  # cfg['model']['encoder'] # LocalPoolPointnet
        dim = 3  # cfg['data']['dim']
        c_dim = 32  # cfg['model']['c_dim']
        padding = 0.1  # cfg['data']['padding']
        decoder_kwargs = \
            {'sample_mode': 'bilinear', 'hidden_size': 32, 'out_dim': 4} # modify here: out_dim
        encoder_kwargs = \
            {'hidden_dim': 32, 'plane_type': ['xz', 'xy', 'yz'], 'plane_resolution': 64, 'unet': True,
             'unet_kwargs': {'depth': 4, 'merge_mode': 'concat', 'start_filts': 32}}

        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding, # modify here dim
            **encoder_kwargs)

        decoder = decoder_dict[decoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **decoder_kwargs
        )

        model = ConvolutionalOccupancyNetwork(
            decoder, encoder, device=device
        ).to(device)

        '''load pretrained weights'''
        pretrain_dict = torch.load('datasets/pretrained/convonet/shapenet_3plane.pt')['model']
        final_dict = {k:v for k,v in pretrain_dict.items() if 'decoder.fc_out' not in k}

        decoder_fc_out_weight = model.state_dict()['decoder.fc_out.weight']
        decoder_fc_out_bias = model.state_dict()['decoder.fc_out.bias']


        decoder_fc_out_weight[0] = pretrain_dict['decoder.fc_out.weight']
        decoder_fc_out_bias[0] = pretrain_dict['decoder.fc_out.bias']
        final_dict['decoder.fc_out.weight'] = decoder_fc_out_weight
        final_dict['decoder.fc_out.bias'] =  decoder_fc_out_bias

        model.load_state_dict(final_dict, strict=True)
        print(f"Loaded pretrained convonet: #params = {sum([p.numel() for p in model.parameters()])}")
        # freeze
        # for param in self.generator.parameters():
        #     param.requires_grad = False
        # self.generator.eval()

        '''load real data'''
        from training.dataset import Dataset
        from torch.utils.data import DataLoader

        test_set = Dataset(split='test', point_per_shape=N)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=B,
                                 shuffle=False)
        i = 0
        for data in (test_loader):
            print(i, '/', len(test_loader))
            # fetch data
            occ_points, occ, voxels, \
            p_coords, p_colors, p_material_ids, p_uvs, \
            mat_images, mat_colors, \
            mat_means_rgb, mat_vars_rgb, mat_means_xyz, mat_vars_xyz, mat_area_ratios, \
            mat_images_ids, mat_colors_ids, p_img_mat_arr_id, p_color_mat_arr_id, mat_cls, names = data
            occ_points = occ_points.to(device)
            occ = occ.to(device)
            p_coords = p_coords.to(device)
            p_colors = p_colors.to(device)

            # print(p_coords.shape)


            '''forward'''
            inputs = p_coords
            print('inputs', inputs.shape)
            c = model.encode_inputs(inputs)  #
            for k, v in c.items():
                print('c', k, v.shape)

            # xz torch.Size([2, 32, 64, 64])
            # xy torch.Size([2, 32, 64, 64])
            # yz torch.Size([2, 32, 64, 64])

            kwargs = {}
            # General points
            logits = model.decode(occ_points, c, **kwargs)#.logits  # B,N
            print('logits.shape', logits.shape)

            '''check sdf'''
            loss_i = F.binary_cross_entropy_with_logits(
                logits[:,:,0], occ, reduction='none')
            # loss = loss_i.sum(-1).mean()
            loss = loss_i.mean()

            accu = (occ.long() == (logits[:,:,0] > 0).long()).float().mean()
            print('loss', loss)
            print('accu', accu)

            '''vis'''
            import trimesh
            from models.networks_get3d import DMTetGeometry

            dmtet_geometry = DMTetGeometry(grid_res=90, scale=1.0, device=device)  # res = 64,70,80,90,100

            positions = dmtet_geometry.verts.unsqueeze(dim=0).expand(B, -1, -1).to(device)
            dmtet_results = model.decode(positions, c, **kwargs)
            occ_logits = sdf = logits = dmtet_results[:,:,0]#.logits  # B,N
            deformation = dmtet_results[:,:,1:]


            v_deformed = positions + deformation
            tets = dmtet_geometry.indices

            v_list = []
            f_list = []
            mesh_list = []
            # Using marching tet to obtain the mesh
            for i_batch in range(B):
                verts, faces = dmtet_geometry.get_mesh(
                    v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                    with_uv=False, indices=tets)
                ################################################################
                from utils.other_utils import rotate_pc, normalize_pc

                verts = normalize_pc(verts)
                # verts = rotate_pc(verts,theta=-0.5*np.pi,axis='y') # N,3
                # print('verts',verts.max().item(),verts.min().item()) # 0.444, -0.456 ...

                ################################################################
                v_list.append(verts)
                f_list.append(faces)

                mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
                vis = True
                if vis:
                    print('i_batch', i_batch)
                    from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk

                    in_points = positions[i_batch][sdf[i_batch] > 0]
                    # in_points = normalize_pc(in_points)
                    # in_points = rotate_pc(in_points, theta=-0.5 * np.pi, axis='y')  # N,3
                    vis_actors_vtk([
                        get_mesh_actor_vtk(mesh, opacity=0.5),
                        # get_colorful_pc_actor_vtk(in_points, opacity=0.9)
                    ], arrows=True)
                mesh_list.append(mesh)

    # quick_test()
    test_pretrained()
    # test_modified()
    # test_dmtet()


