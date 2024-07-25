import sys
sys.path.append("..")
sys.path.append(".")
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

import nvdiffrast.torch as dr

from models.TextureField.convonet import ConvolutionalOccupancyNetwork,LocalDecoder,LocalPoolPointnet


# import trimesh
# import kaolin as kal
# from utils.other_utils import increase_p_num, make_3d_grid,pad_marching_cubes,make_3d_grid_batch


class Network(torch.nn.Module):
    def __init__(self,
            
                 device = 'cuda',
                 decoder_only = False,
                 img_resolution = 1024,
                 img_feat_dim = 200

                 ):
        super().__init__()
   
        self.device = device

        # render
        self.img_resolution = img_resolution
        self.img_feat_dim = img_feat_dim
        self.decoder_only = decoder_only
        # try:
        #     self.ctx = dr.RasterizeGLContext(device=self.device)
        # except:
        #     self.ctx = dr.RasterizeCudaContext(device=self.device)#
        self.ctx = dr.RasterizeCudaContext(device=self.device)#

        dim = 3  # cfg['data']['dim']
        c_dim = 32  # cfg['model']['c_dim']
        padding = 0.1  # cfg['data']['padding']
        
        
        
        if decoder_only:
            encoder = None
        else:
            texture_encoder_kwargs = \
                {'hidden_dim': 32, 'plane_type': ['xz', 'xy', 'yz'], 'plane_resolution': 64, 'unet': True,
                 'unet_kwargs': {'depth': 4, 'merge_mode': 'concat', 'start_filts': 32}}
            encoder = LocalPoolPointnet(
            dim=6, c_dim=c_dim, padding=padding,  # modify here: dim =6: xyzrgb
            **texture_encoder_kwargs)

        decoder_kwargs = \
            {'sample_mode': 'bilinear', 'hidden_size': 32, 'out_dim': 3}  # modify here: add out_dim =3 for rgb

        decoder = LocalDecoder(
            dim=dim, c_dim=c_dim, padding=padding,
            **decoder_kwargs
        )
        self.rgb_net = ConvolutionalOccupancyNetwork(decoder=decoder,encoder=encoder, device=device).to(device)


    '''network predictions'''
   
    def encode_rgb_feature(self,p_features):
        # p_features = torch.cat((p_coords,p_colors),2)
        c = self.rgb_net.encode_inputs(p_features)  # dict: 'xz','xy','yz': torch.Size([B, 32, 64, 64])
        return c


    def pred_rgb(self,c,positions):
        # p_features = torch.cat((p_coords,p_colors),2)
        # c = self.tex_net.encode_inputs(p_features)  # dict: 'xz','xy','yz': torch.Size([B, 32, 64, 64])
        kwargs = {}
        rgbs = self.rgb_net.decode(positions, c, **kwargs)#.logits  # B,N

        return rgbs



    def query_img_pixel(self,uv,image,h=256,w=256):
        '''
        :param uv: [num_images,num_points,2]
        :param image: [num_images,C,H,W]   RGB 0-1
        :param h:
        :param w:
        :return: [num_images,num_points,3] RGB -1~1
        '''
        # print('devices',image.device,uv.device)
        self.img_pix_color_queryer.eval()
        self.img_pix_color_queryer.gen_feat(((image - 0.5) / 0.5)) # make the image pixel values from -1 to 1
        coord = uv
        cell = torch.ones_like(coord).to(coord.device)

        cell[:,:, 0] *= 2 / h
        cell[:,:, 1] *= 2 / w

        pred = self.img_pix_color_queryer.query_rgb(coord, cell)

        # pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        return pred # random range but supervize it to be -1 to 1
   

  
    # texture field
    def get_textured_mesh(self,p_coords,p_colors,names,save_root,use_GT_geo = False,
                          use_existing_geo = False,mesh_v=None, mesh_f=None,xatlas_root_path = None):
        '''
        Based on 'extract_3d_shape' of GET3D
        :return:
        '''
        # Step 1: predict geometry first (pass)
  
        # Step 2: use x-atlas to get uv mapping for the mesh (slow)
        print('Step 2: use x-atlas to get uv mapping for the mesh')
        from models.get3d.extract_texture_map import xatlas_uvmap
        all_uvs = []
        all_mesh_tex_idx = []
        all_gb_pose = []
        all_uv_mask = []
        if self.ctx is None:
            print('self.ctx is None')
            try:
                self.ctx = dr.RasterizeGLContext(device=self.device)
            except:
                self.ctx = dr.RasterizeCudaContext(device=self.device)#
        print('start for v, f in zip(mesh_v, mesh_f):') # slow slow slow
        i =-1
        for v, f in zip(mesh_v, mesh_f): # slow
            i = i+1
            if xatlas_root_path is not None :
                xatlas_save_path = (os.path.join(xatlas_root_path,names[i],'xatlas.pth'))
                if os.path.exists(xatlas_save_path): # load
                    xatlas_dict = torch.load(xatlas_save_path)
                    uvs = xatlas_dict['uvs']
                    mesh_tex_idx = xatlas_dict['mesh_tex_idx']
                    gb_pos = xatlas_dict['gb_pos']
                    mask = xatlas_dict['mask']
                else: # calculate and save
                    uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(
                        self.ctx, v, f, resolution=self.img_resolution)
                    xatlas_dict = {'uvs':uvs,'mesh_tex_idx':mesh_tex_idx,'gb_pos':gb_pos,'mask':mask}
                    os.makedirs(os.path.dirname(xatlas_save_path),exist_ok=True)
                    torch.save(xatlas_dict,xatlas_save_path)
            else: # calculate without saving
                uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(
                    self.ctx, v, f, resolution=self.img_resolution)

            all_uvs.append(uvs)
            all_mesh_tex_idx.append(mesh_tex_idx)
            all_gb_pose.append(gb_pos)
            all_uv_mask.append(mask)


        tex_hard_mask = torch.cat(all_uv_mask, dim=0).float() # batch,res,res,1
        print('tex_hard_mask.shape',tex_hard_mask.shape)

        # Step 3: Query the texture field to get the RGB color for texture map
        print('Step 3: Query the texture field to get the RGB color for texture map')
        # we use run one per iteration to avoid OOM error
        all_network_output = []
        p_features = torch.cat((p_coords, p_colors), 2)

        i=-1
        for _p_features, _all_gb_pose , _tex_hard_mask in zip(p_features, all_gb_pose, tex_hard_mask):
            i = i+1
            # print('_all_gb_pose.shape',_all_gb_pose.shape) # [1. img_res,img_res,3]
            _all_gb_pose = _all_gb_pose.reshape(1,-1,3) # [1. img_res*img_res,3]

            c_rgb = self.encode_rgb_feature(_p_features.unsqueeze(0))
            rgb = self.pred_rgb(c_rgb,_all_gb_pose) # [1. img_res*img_res,3]
            rgb =  rgb.reshape(1,self.img_resolution,self.img_resolution,3) # [1. img_res,img_res,3]
            # rgb *= tex_hard_mask[i].unsqueeze(0) # added by Qiao. don't put it here, for it will *2-1 later
            all_network_output.append(rgb)
            # print('rgb.shape',rgb.shape) # [1. img_res,img_res,3]
        network_out = torch.cat(all_network_output, dim=0)

        generated_mesh = mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out,names
        # Step 4: save results
        from models.get3d.get3d_utils.utils_3d import save_obj, savemeshtes2
        import PIL.Image
        import cv2

        i = -1
        for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map,name in zip(*generated_mesh):
            i = i + 1
            if '/' in name:
                save_path = os.path.join(save_root,f'{name.split("/")[1]}.png')
            else:
                save_path = os.path.join(save_root,name,'models', f'model_normalized.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            savemeshtes2(
                mesh_v.data.cpu().numpy(),
                all_uvs.data.cpu().numpy(),
                mesh_f.data.cpu().numpy(),
                all_mesh_tex_idx.data.cpu().numpy(),
                # save_path.replace('.png','.obj')
                os.path.join(save_root,name,'models', 'model_normalized.obj')
            )
            lo, hi = (-1, 1)
            print('tex_map.shape',tex_map.shape)
            # img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32) # don't permute, it's WHC already
            img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            img *= tex_hard_mask[i].detach().cpu().numpy() # added by Qiao, necessary to enable later dialate
            img = img.clip(0, 255)
            mask = np.sum(img.astype(float), axis=-1,
                          keepdims=True)  # mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
            mask = (mask <= 3.0).astype(float)  # mask = (mask <= 3.0).astype(np.float)
            kernel = np.ones((3, 3), 'uint8')
            dilate_img = cv2.dilate(img, kernel, iterations=1) # without this, some faces will have edges with wrong colors
            img = img * (1 - mask) + dilate_img * mask
            img = img.clip(0, 255).astype(np.uint8)


            PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                save_path)
            # save_mesh_idx += 1

    

if __name__ == '__main__':
    def quick_test():
        # basic configs
        device = torch.device('cuda', 0)
        B = 5
        N = 2048
        p_coords = torch.randn([B, N, 3], device=device)
        p_colors = torch.randn([B, N, 3], device=device)
       
       
        p_features = torch.cat((p_coords, p_colors), 2)

        # with encoder
        # model = Network(device=device).train().requires_grad_(False).to(device)
        # rgb_feature = model.encode_rgb_feature(p_features) # dict,'xz','xy','yz': torch.Size([B, 32, 64, 64])
        # print('rgb_feature.shape', type(rgb_feature))
        # pred_rgb = model.pred_rgb(rgb_feature, p_coords)
        # print('pred_rgb.shape', pred_rgb.shape) # B,N,3

        # without encoder
        model = Network(device=device,decoder_only=True).train().requires_grad_(False).to(device)
        rgb_feature = {'xz': torch.randn([B, 32, 64, 64], device=device), 
                       'xy': torch.randn([B, 32, 64, 64], device=device), 
                       'yz': torch.randn([B, 32, 64, 64], device=device)}
        pred_rgb = model.pred_rgb(rgb_feature, p_coords)
        print('pred_rgb.shape', pred_rgb.shape) # B,N,3



    quick_test()




