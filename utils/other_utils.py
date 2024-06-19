import sys
sys.path.append('..')
sys.path.append('.')
import torch
import trimesh
import mcubes
import numpy as np
import math
from plyfile import PlyData, PlyElement
import PIL
import os
# import open3d as o3d
from torchvision.transforms import transforms
from scipy.sparse import coo_matrix



def increase_p_num(src, N2):
    '''
    :param src: torch tensor of shape [N1,C]
    :param N2: N2 >= N1
    :return: torch tensor of shape [N2,C]
    '''
    N1, C = src.shape
    device = src.device
    assert N2 >= N1
    target = torch.zeros((N2, C), device=device)
    target[:N1] = src
    if N2 > N1:

        target[N1:] = src[torch.randint(N1, (N2 - N1,), dtype=torch.long)]
    return target

def pad_points(src, N2,value=0):
    '''

    :param src: torch tensor of shape [N1,C]
    :param N2: N2 >= N1
    :param value: a single number. use this value to pad
    :return: torch tensor of shape [N2,C]
    '''
    N1, C = src.shape
    return torch.cat((src,value * torch.ones((N2-N1,C),device = src.device)),0)

def make_3d_grid(min = -0.5,max = 0.5,resolution = 32):
    '''
    :param min: min coordinates of the grid
    :param max:
    :param resolution:
    :return: p: a torch tensor on cpu of shape: resolution^3, 3
    '''
    bb_min = (min,) * 3
    bb_max = (max,) * 3
    shape = (resolution,) * 3

    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def make_3d_grid_batch(batch_size=1,min = -0.5,max = 0.5,resolution = 32):
    p = make_3d_grid(min,max,resolution)
    p = p.unsqueeze(0)
    p = p.repeat(batch_size, 1, 1)
    return p

def pad_marching_cubes(occ_logits, padding=0.1, threshold=0.5):
    '''
    Code from Occupancy Networks.
    Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_logits (np): value grid of occupancies. [resolution,resolution,resolution]
        padding:
        threshold: assume the occ_hat will be sigmoid, the threshold after sigmoid.
    '''

    # nx = resolution
    # pointsf = box_size * make_3d_grid(
    #     (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
    # )
    # values = self.eval_points(pointsf, z, c, device, teacher=teacher, **kwargs).cpu().numpy()
    # value_grid = values.reshape(nx, nx, nx)

    # occ_hat = value_grid
    # Some short hands

    if torch.is_tensor(occ_logits):
        occ_logits = occ_logits.detach().cpu().numpy()

    n_x, n_y, n_z = occ_logits.shape
    box_size = 1 + padding
    threshold = np.log(threshold) - np.log(1. - threshold) # a
    # Make sure that mesh is watertight

    occ_hat_padded = np.pad(
        occ_logits, 1, 'constant', constant_values=-1e6)

    vertices, triangles = mcubes.marching_cubes(
        occ_hat_padded, threshold)
    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)
    triangles = np.array(triangles).astype(np.int64)
    # # Create mesh
    # mesh = trimesh.Trimesh(vertices, triangles,process=False)
    # return mesh
    return vertices,triangles

def save_colored_pc_ply(coords,colors, path):
    '''

    :param coords:
    :param colors: float within range of 0-1
    :param path:
    :return:
    '''

    # colors *=255
    vertices = np.empty(coords.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')
                               ])
    vertices['x'] = coords[:, 0].astype('f4')
    vertices['y'] = coords[:, 1].astype('f4')
    vertices['z'] = coords[:, 2].astype('f4')

    vertices['red'] = colors[:, 0].astype('f4') *255
    vertices['green'] = colors[:, 1].astype('f4') *255
    vertices['blue'] = colors[:, 2].astype('f4') *255

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)

    ply.write(path)

def read_ply_xyz(file):
    ply = PlyData.read(file)
    vtx = ply['vertex']

    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    return xyz

def read_ply_xyzrgb(file):
    ply = PlyData.read(file)
    vtx = ply['vertex']

    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    rgb = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=-1)

    return xyz,rgb


def get_rotation_matrix (theta, axis):
    if axis == 'x':
        rotation_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]])
    elif axis == 'y':
        rotation_matrix = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        rotation_matrix = np.array(
            [[np.cos(theta), np.sin(-theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]])
    else:
        print('wrong axis:',axis)
        rotation_matrix = 'error'
    return rotation_matrix

def rotate_pc(pc,theta = 0.5*np.pi,axis='y'):
    '''
    No batch
    :param pc: torch.float, [ N, 3]
    :param theta: something like -0.5 * np.pi
    :param axis: 'x', 'y', or 'z'
    :return:
    '''
    axis_rectified = get_rotation_matrix(theta, axis)
    axis_rectified = torch.tensor(axis_rectified).to(pc.device).float()
    pc = torch.matmul(pc, axis_rectified)
    return pc

def normalize_pc(pc):
    '''
    :param pc:
    :return:
    '''
    batch = True
    if pc.dim() == 2:
        pc = pc.unsqueeze(0)
        batch = False
    coords_max = pc.max(1)[0] # [0] if tensor
    coords_min = pc.min(1)[0]
    coords_center = (coords_max + coords_min) / 2
    coords_size = coords_max - coords_min
    pc -= coords_center
    pc /= coords_size.max(1)[0]
    if not batch:
        pc = pc.squeeze(0)
    return pc

def rescale_pc(pc):
    min_xyz = torch.min(pc, 1)[0]

    max_xyz = torch.max(pc, 1)[0]
    lengh_xyz = max_xyz - min_xyz
    scale = torch.max(lengh_xyz, 1)[0]  #
    pc = pc.transpose(0, 2)
    pc /= scale
    pc = pc.transpose(0, 2)



def apply_texture2geo(v_list,f_list,uv_list,f_uv_idx_list, \
                      f_mat_id_list,
                      mat_images_ids, mat_colors_ids,mat_images,mat_colors,name,
                      save_root = 'out/temp'):
    '''
    reference: GET3D, utils_3d.py, savemeshtes2
    :param v_list:
    :param f_list:
    :param uv_list:
    :param f_mat_id_list:
    :param mat_images_ids:
    :param mat_colors_ids:
    :param mat_images:
    :param mat_colors:
    :param name:
    :return:
    '''
    from training.dataset import total_mat_id2mat_array_id
    import PIL.Image


    # cls,name = name.split('/')
    os.makedirs(os.path.join(save_root,name,"models"),exist_ok=True)
    obj_file_name = os.path.join(save_root,name,"models","model_normalized.obj")
    mtl_file_name = os.path.join(save_root,name,"models","model.mtl")

    verts = v_list #pointnp_px3
    v_uvs = uv_list # tcoords_px2
    faces = f_list # v indices in the face # facenp_fx3
    facetex_fx3 = f_uv_idx_list  # vt indices in the face # v and vt have 1to1 indices # facetex_fx3
    # facetex_fx3 = f_list[i] # vt indices in the face # v and vt have 1to1 indices # facetex_fx3
    f_raw_mat_ids = f_mat_id_list # facemat_f
    # all_mat_ids = f_raw_mat_ids.unique()

    # Step 1: write .obj file
    with open(obj_file_name,'w') as file:
        # file = open(obj_file_name, 'w')
        file.write('mtllib model.mtl\n' )

        for pidx, p in enumerate(verts):
            pp = p
            file.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

        for pidx, p in enumerate(v_uvs):
            pp = p
            file.write('vt %f %f\n' % (pp[0], pp[1]))

        for mat_id in mat_images_ids:
            if mat_id >-100:
                file.write(f'usemtl material_{int(mat_id)}\n')
                mask = f_raw_mat_ids == mat_id
                # print('f_raw_mat_ids.shape',f_raw_mat_ids.shape)
                # print('faces.shape',faces.shape)
                selected_face_index = np.arange(faces.shape[0])[mask]
                selected_faces = faces[mask]
                # for fidx, f in enumerate(facenp_fx3):
                for fidx, selected_face in zip(selected_face_index,selected_faces):
                    f1 = selected_face + 1
                    f2 = facetex_fx3[fidx] + 1
                    file.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
        for mat_id in mat_colors_ids:
            if mat_id >-100:
                file.write(f'usemtl material_{int(mat_id)}\n')
                mask = f_raw_mat_ids == mat_id
                selected_face_index = np.arange(faces.shape[0])[mask]
                selected_faces = faces[mask]
                # for fidx, f in enumerate(facenp_fx3):
                for fidx, selected_face in zip(selected_face_index, selected_faces):
                    f1 = selected_face + 1
                    f2 = facetex_fx3[fidx] + 1
                    file.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))


    # Step 2: write .mtl file
    with open(mtl_file_name,'w') as file:
        for mat_id in mat_images_ids:
            if mat_id>-100:
                mat_id = int(mat_id)
                file.write(f'newmtl material_{int(mat_id)}\n')
                file.write('Kd 1 1 1\n')
                file.write('Ka 0 0 0\n')
                file.write('Ks 0.4 0.4 0.4\n')
                file.write('Ns 10\n')
                file.write('illum 2\n')
                file.write('map_Kd %s.png\n' % mat_id)

        for mat_id in mat_colors_ids:
            if mat_id >-100:
                # mat_id = int(mat_id)
                file.write(f'newmtl material_{int(mat_id)}\n')
                # file.write('Kd 1 1 1\n')
                color = mat_colors[total_mat_id2mat_array_id(mat_id,mat_colors_ids)]
                file.write(f'Kd {color[0]} {color[1]} {color[2]}\n')
                file.write('Ka 0 0 0\n')
                file.write('Ks 0.4 0.4 0.4\n')
                file.write('Ns 10\n')
                file.write('illum 2\n')
                # file.write('map_Kd %s.png\n' % mat_id)

    # Step 3: save .png file
    for mat_id in mat_images_ids:
        if mat_id>-100:
            img = mat_images[total_mat_id2mat_array_id(mat_id,mat_images_ids)]
            # print('img',img.shape)
            lo, hi = (0, 1)
            # img = np.asarray(img.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
            img = np.asarray(img.transpose(1, 2, 0), dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            img = img.clip(0, 255)


            img = img.clip(0, 255).astype(np.uint8)
            PIL.Image.fromarray(np.ascontiguousarray(img), 'RGB').save(os.path.join(save_root,name,"models",f"{int(mat_id)}.png"))




# -----------------------------------------
def get_multiview_pc_validation(pc_np,cam_locations,hidden_point_removal_radius=100.0):
    import open3d as o3d
    point_visibility = np.zeros((len(cam_locations),len(pc_np))).astype(np.bool_)
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_np))
    for i,cam_location in enumerate(cam_locations):
        # o3d_camera = [0, 0, diameter]
        o3d_camera = np.array(cam_location)

        _, pt_map = pcd.hidden_point_removal(o3d_camera, hidden_point_removal_radius)


        visible_point_ids = np.array(pt_map)
        point_visibility[i,visible_point_ids] = True
    return point_visibility


# --------------------------------------test stuff --------------------------

def run_apply_texture2geo(use_chess_texture = False,save_root = 'out/temp',norm_texture_uv = True,by_scale = True):
    import kaolin as kal
    from training.dataset import total_mat_id2mat_array_id #
    shapenet_clean_root_path = '/home/me/RfDNet/datasets/ShapeNetCore.v2.clean_mesh'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda', 0)
    B = 5
    N = 2048
    latent_dim = 512
    # model = Network(dmtet_scale=1.0, device=device).train().requires_grad_(False).to(device)
    '''load real data'''

    from training.dataset import Dataset
    from torch.utils.data import DataLoader

    test_set = Dataset(split='test', point_per_shape=N,norm_texture_n_uv=norm_texture_uv)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=B,
                             shuffle=False)
    chess_img = torch.tensor(np.array(PIL.Image.open('datasets/chess.png')))  # H,W,C = 4
    chess_img = chess_img.permute(2,0,1)
    chess_img = chess_img[:3,:,:]
    chess_img = transforms.Resize((test_set.mat_img_size, test_set.mat_img_size))(chess_img)  # [3,mat_img_size,mat_img_size]
    chess_img = chess_img.numpy()/255.0
    # print('chess_img.shape', chess_img.shape)


    i = 0
    for data in (test_loader):
        print(i, '/', len(test_loader))
        # fetch data
        occ_points, occ, voxels, \
        p_coords, p_colors, p_material_ids, p_uvs, \
        mat_images, mat_colors, \
        mat_means_rgb, mat_vars_rgb, mat_means_xyz, mat_vars_xyz, mat_area_ratios, \
        mat_images_ids, mat_colors_ids, p_img_mat_arr_id, p_color_mat_arr_id, mat_cls, names = data

        for i_batch,name in enumerate(names):
            mesh_file_name = os.path.join(shapenet_clean_root_path,  name, 'models', 'model_normalized.obj')
            mesh = kal.io.obj.import_mesh(mesh_file_name, with_materials=True)

            # load gt mesh
            mesh_v = mesh.vertices  # num_points,3
            mesh_f = mesh.faces
            uvs = mesh.uvs

            face_uvs_idx = mesh.face_uvs_idx

            # get face_material_id
            materials_order = mesh.materials_order
            # uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0),(0, 0, 0, 1))
            nb_faces = mesh_f.shape[0]
            num_consecutive_materials = \
                torch.cat([
                    materials_order[1:, 1],
                    torch.LongTensor([nb_faces])
                ], dim=0) - materials_order[:, 1]
            face_material_idx = kal.ops.batch.tile_to_packed(
                materials_order[:, 0],
                num_consecutive_materials
            ).squeeze(-1)
            # get images etc.
            mat_images_ids_ = mat_images_ids.detach().cpu().numpy()[i_batch]
            mat_colors_ids_ = mat_colors_ids.detach().cpu().numpy()[i_batch]
            mat_images_ = mat_images.detach().cpu().numpy()[i_batch]
            mat_colors_ = mat_colors.detach().cpu().numpy()[i_batch]


            if norm_texture_uv:
                uvs[:, 1] = -uvs[:, 1]
                for mat_img_arr_id, mat_id in enumerate(mat_images_ids[i_batch]):
                    if mat_id > -100:
                        mask = face_uvs_idx[face_material_idx==mat_id] # ?,3
                        mask = mask.reshape(-1) # ?
                        # print(mask.shape, 'mask.shape')
                        ori_img = mat_images[i_batch][mat_img_arr_id]
                        ori_uvs = uvs[mask]  # [?,2]
                        if mat_id ==0:
                            print('ori_uvs',ori_uvs.shape)
                            u_max, v_max = ori_uvs.max(0)[0]
                            u_min, v_min = ori_uvs.min(0)[0]
                            print( u_max, v_max, u_min, v_min)
                            print('ori_uvs',ori_uvs)
                        # print('ori_uvs.shape',ori_uvs.shape) # ?,2

                        if by_scale:
                            if len(mask) >0:
                                # temp = p_uvs[i_batch][p_material_ids[i_batch] == mat_id]
                                # print('temp.shape',temp.shape)
                                # u_max, v_max = p_uvs[i_batch][p_material_ids[i_batch] == mat_id].max(0)
                                # u_min, v_min = p_uvs[i_batch][p_material_ids[i_batch] == mat_id].min(0)
                                # print('min max',u_max,v_max,u_min,v_min)

                                u_max, v_max = ori_uvs.max(0)[0]
                                u_min, v_min = ori_uvs.min(0)[0]

                                # get new_uv
                                new_uvs = ori_uvs  # .clone()
                                new_uvs[:, 0] = (new_uvs[:, 0] - u_min) / (u_max - u_min)
                                new_uvs[:, 1] = (new_uvs[:, 1] - v_min) / (v_max - v_min)
                                uvs[mask] = new_uvs
                        if mat_id == 0:
                            print('new_uvs',new_uvs.shape)
                            u_max, v_max = new_uvs.max(0)[0]
                            u_min, v_min = new_uvs.min(0)[0]
                            print(u_max, v_max, u_min, v_min)
                            print('new_uvs' ,new_uvs)
                uvs[:, 1] = -uvs[:, 1]

            if use_chess_texture:
                # # get face_material_id
                # face_material_idx = torch.zeros((mesh_f.shape[0]))
                # # get images etc.
                # mat_images_ids_ = np.ones((test_set.max_image_material)) * -100 #np.array([0,-100,-100,-100,-100])
                # mat_images_ids_[0] = 0
                # mat_colors_ids_ = np.ones((test_set.max_color_material)) * -100
                # mat_images_ = np.zeros((test_set.max_image_material,3,test_set.mat_img_size,test_set.mat_img_size))
                # mat_colors_ = np.array((test_set.max_color_material,3))

                mat_images_[0] = chess_img
                mat_images_[1] = chess_img
                mat_images_[2] = chess_img
                mat_images_[3] = chess_img
                mat_images_[4] = chess_img


            # normalize face_uvs_idx
            mask = face_uvs_idx == -1
            face_uvs_idx[mask] = 0




            # print('mesh_f.shape',mesh_f.shape) #
            # print('face_material_idx.shape',face_material_idx.shape)
            # # print('face_uvs.shape',face_uvs.shape)
            # print('face_uvs_idx.shape',face_uvs_idx.shape)
            # print('mesh.uvs.shape',mesh.uvs.shape)
            # print('-'*50)

            mesh_v = mesh_v.detach().cpu().numpy()
            mesh_f = mesh_f.detach().cpu().numpy()
            mesh_uv =  np.array(uvs)
            face_uvs_idx = face_uvs_idx.detach().cpu().numpy()
            face_material_idx = face_material_idx.detach().cpu().numpy()




            apply_texture2geo( v_list = mesh_v , f_list=mesh_f,uv_list = mesh_uv,
                               f_uv_idx_list = face_uvs_idx, f_mat_id_list = face_material_idx,
                               mat_images_ids=mat_images_ids_,mat_colors_ids=mat_colors_ids_,
                               mat_images=mat_images_, mat_colors = mat_colors_,
                               name=name,save_root=save_root)

        i = i+1

def generate_chessed_shapes():
    pass

def test_dmtet_verts():
    from models.network import Network
    # https://math.stackexchange.com/questions/2576824/decompose-a-cube-into-tetrahedra-more-than-one-way
    device = torch.device('cuda',0)
    model = Network(dmtet_scale=1.0, device=device).train().requires_grad_(False).to(device)
    v_deformed = model.dmtet_geometry.verts#.unsqueeze(dim=0)#.expand(sdf.shape[0], -1, -1) + deformation
    tets = model.dmtet_geometry.indices
    sdf = torch.ones((v_deformed.shape[0])).to(device).float()
    print('dmtet',v_deformed.shape,tets.shape)
    # print(v_deformed[:20])
    # print()

    # sdf [1000:1100] = -1.0
    # print('v_deformed.shape',v_deformed.shape)
    # print('sdf.shape',sdf.shape)

    # verts, faces = model.dmtet_geometry.get_mesh(v_deformed, sdf, with_uv=False, indices=tets)
    # mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(),faces=faces.detach().cpu().numpy())

    save_tet_grid = False
    if save_tet_grid:
        # save whole tet grid
        mesh = trimesh.Trimesh(vertices=v_deformed.detach().cpu().numpy(),faces=tets.detach().cpu().numpy())
        # mesh.export('tet.obj')
        from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk, get_pc_actor_vtk
        vis_actors_vtk([
            get_mesh_actor_vtk(mesh, opacity=1),  # pred_mesh
            # get_colorful_pc_actor_vtk(v_deformed [1000:1100] ,opacity = 0.9,point_size=10)

        ], arrows=True)
        # mesh.export('tet.obj')

    get_my_tet_grid = True
    if get_my_tet_grid:
        import numpy as np

        # Define the number of cubes in each dimension
        res = 32
        length = 1.0/res
        # Define the vertices of the 1x1 cube
        cube_verts = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],

        ])
        tet_indices_base= np.array([
            [1,3,7,2],
            [4,6,7,2],
            [1,4,2,7],
            [1,4,0,2],

        ])

        # mesh = trimesh.Trimesh(vertices=cube_verts, faces=tet_indices_base)
        # mesh.export('my_tet_single_cube.obj')
        # from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk, get_pc_actor_vtk
        # vis_actors_vtk([
        #     get_mesh_actor_vtk(mesh, opacity=1),  # pred_mesh
        #     # get_colorful_pc_actor_vtk(verts ,opacity = 0.9,point_size=3)
        #
        # ], arrows=True)

        # Divide the cube into res x res x res cubes
        # x = np.linspace(0, 1, res + 1)
        # y = np.linspace(0, 1, res + 1)
        # z = np.linspace(0, 1, res + 1)
        x = np.arange(0,  res + 1)
        y = np.arange(0, res + 1)
        z = np.arange(0, res + 1)

        grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3) #+ 0.5 / res
        # print('grid_points.shape',grid_points.shape,grid_points) # res^3,3


        cube_centers = np.array(np.meshgrid(x[:-1], y[:-1], z[:-1])).T.reshape(-1, 3) #+ 0.5 / res
        # print('cube_centers.shape',cube_centers.shape) # res^3,3

        # Define the vertices and indices of the tetrahedra
        tet_verts = []
        tet_indices = []
        for cx, cy, cz in cube_centers:
            # Define the vertices of the current cube
            cube_verts_shifted = cube_verts + np.array([cx, cy, cz])

            # Define the indices of the tetrahedra within the current cube
            # tet_indices_shifted = np.array([
            #     [0, 1, 2, 3],
            #     [1, 4, 5, 3],
            #     [1, 5, 7, 4],
            #     [1, 7, 6, 5],
            #     [1, 3, 7, 2],
            #     [3, 6, 7, 2],
            #     [0, 3, 2, 6],
            # ]) + len(tet_verts)
            tet_indices_shifted = tet_indices_base + len(tet_verts)*8

            # Add the vertices and indices of the tetrahedra to the global list
            tet_verts.append(cube_verts_shifted)
            tet_indices.append(tet_indices_shifted)

        tet_verts = np.concatenate(tet_verts, axis=0)
        tet_indices = np.concatenate(tet_indices, axis=0)
        # print(tet_verts.shape, tet_indices.shape, tet_indices[:20])
        # Ensure that the vertices are unique (there may be some overlap between adjacent cubes)

        tet_verts, unique_idx = np.unique(tet_verts, axis=0, return_inverse=True)
        # print('unique_idx',unique_idx.shape,tet_verts.shape)
        # print('unique_idx',unique_idx)
        tet_indices = unique_idx[tet_indices]# tet_indices = np.searchsorted(unique_idx, tet_indices)  # problem is here
        # print(tet_verts.shape,tet_indices.shape,tet_indices.max())

        # Ensure that the indices are sorted in ascending order
        tet_indices = np.sort(tet_indices, axis=1)
        # print(tet_verts.shape, tet_indices.shape,tet_indices[:20])

        # Remove duplicate tetrahedra
        tet_indices, unique_idx = np.unique(tet_indices, axis=0, return_index=True)

        # Define the vertices of the resulting tetrahedra and their indices
        verts = tet_verts
        tets = tet_indices

        print('-'*50)
        print('my tet',verts.shape,tets.shape)

        verts = verts * 1.0 / float(res) -0.5

        print(verts.max(), verts.min())

        np.savez('my_tet.npz',vertices = verts,tets=tets)
        print('saved')

        mesh = trimesh.Trimesh(vertices=verts, faces=tets)
        mesh.export('my_tet.obj')
        from utils.vtk_basic import vis_actors_vtk, get_mesh_actor_vtk, get_colorful_pc_actor_vtk, get_pc_actor_vtk
        vis_actors_vtk([
            get_mesh_actor_vtk(mesh, opacity=1),  # pred_mesh
            get_colorful_pc_actor_vtk(verts ,opacity = 0.9,point_size=3)

        ], arrows=True)

def get_face_adjacency_matrix(mesh):
    '''

    :param mesh: trimesh mesh
    :return:
    '''
    # Calculate the face adjacency
    face_adjacency = mesh.face_adjacency

    # Create a sparse adjacency matrix
    rows = np.concatenate((face_adjacency[:, 0], face_adjacency[:, 1]))
    cols = np.concatenate((face_adjacency[:, 1], face_adjacency[:, 0]))
    data = np.ones(rows.shape, dtype=int)
    adjacency_matrix = coo_matrix((data, (rows, cols)), shape=(len(mesh.faces), len(mesh.faces)))

    # Convert the sparse matrix to a dense matrix if needed
    adjacency_matrix_dense = adjacency_matrix.toarray()
    return adjacency_matrix_dense

def my_smoothing_mesh(vertices,triangles):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    '''simplify'''
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=10000) # 6500
    # mesh_out = mesh_smp
    # mesh_smp = mesh
    '''Smooth by Taubin filter'''
    mesh_out = mesh_smp.filter_smooth_taubin(number_of_iterations=5) # 10
    # mesh_out.compute_vertex_normals()

    '''delete invalid stuff'''
    vertices = np.asarray(mesh_out.vertices)
    triangles = np.asarray(mesh_out.triangles)

    point_nan_mask = np.isnan(vertices).any(axis=1) # [N], point validation
    triangle_nan_mask =  np.isnan(vertices[triangles]).reshape(len(triangles),-1).any(axis=1) # [M,3,3] -> [M,9] -[M] # not working

    # triangle_nan_mask = triangles[point_nan_mask].any(axis=1) # [M,3] ->[M],
    # print('triangle_nan_mask.shape',triangle_nan_mask.shape)
    # print('triangle_nan_mask.sum',triangle_nan_mask.astype(np.int32).sum())
    # print('vertices has nan',np.isnan(vertices).shape) #False
    # print('triangles has nan',vertices[triangles].shape) #False
    mesh_out.remove_triangles_by_mask(triangle_nan_mask)
    mesh_out.remove_unreferenced_vertices()

    vertices = np.asarray(mesh_out.vertices)
    triangles = np.asarray(mesh_out.triangles)
    # print('vertices has nan',np.isnan(vertices).any())

    return vertices,triangles,mesh_out


if __name__ == '__main__':
    # p = make_3d_grid_batch(batch_size=5)
    # print(p.shape,p)

    # import torch
    #
    # a = torch.randn(7, 8, 9)
    # print("Number of dimensions:", a.dim())
    run_apply_texture2geo(use_chess_texture=False, save_root='out/temp')
    # run_apply_texture2geo(use_chess_texture=True,save_root='datasets/ShapeNetCore.v2.clean_mesh_chess')
