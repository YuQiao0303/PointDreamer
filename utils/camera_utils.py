import os
import math
import numpy as np
import torch
import nvdiffrast
import nvdiffrast.torch as dr
import kaolin as kal

import imageio
from PIL import Image #
import sys
import datetime
import pytz # time zone
from utils.utils_2d import cat_images,display_CHW_RGB_img_np_matplotlib
from utils.logger_util import get_logger
import traceback
# device = torch.device('cuda')
# glctx = nvdiffrast.torch.RasterizeGLContext(False, device='cuda') #


def elevation_azimuth_radius_to_xyz(elevation, azimuth,radius):
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    camera_position = np.array([x, y, z])
    return camera_position

def init_6_canical_cams(res,device):
    '''
                  Y
                   ^
                   |
                   |---------> X
                  /
                Z
       '''
    cams = [
        # front
        kal.render.camera.Camera.from_args(eye=torch.tensor([0., 0., -1.5]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),
        # back
        kal.render.camera.Camera.from_args(eye=torch.tensor([0., 0., 1.5]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([0., -1.5, 0.]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 0, 1.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([0., 1.5, 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 0., 1.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),
        # left
        kal.render.camera.Camera.from_args(eye=torch.tensor([-1.5, 0., 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),

        kal.render.camera.Camera.from_args(eye=torch.tensor([1.5, 0., 0]),
                                           at=torch.tensor([0., 0., 0.]),
                                           up=torch.tensor([0., 1., 0.]),
                                           fov=math.pi * 45 / 180,
                                           width=res, height=res, device=device),
    ]

    base_dirs = torch.tensor([
        [0, 0, -1.0],
        [0, 0, 1.0],
        [0, -1.0, 0],
        [0, 1.0, 0],
        [-1.0, 0, 0],
        [1.0, 0, 0],
    ], dtype=torch.float, device=device)
    return cams,base_dirs

def fibonacci_sphere(samples, radius):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_y = math.sqrt(1 - y*y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_y * radius
        z = math.sin(theta) * radius_y * radius
        y = y * radius

        points.append((x, y, z))

    return np.array(points)

def calculate_up_vector(eye_position, target_position,world_up = None):
    gaze_direction = target_position - eye_position
    if world_up is None:
        world_up = np.array([0, 1, 0])
    if np.allclose(np.cross(gaze_direction, world_up), 0):
        up_vector = np.array([0, 0, 1])
    else:
        side_vector = np.cross(gaze_direction, world_up)
        up_vector = np.cross(side_vector, gaze_direction)
        up_vector /= np.linalg.norm(up_vector)
    return up_vector

def create_cameras(num_views=8, distance =1.6,res =512,
                   distribution = 'fibonacci_sphere',
                   device = torch.device('cuda'),vis = False):
    '''

    :param num_views: number of views
    :param D: distance to the origin
    :return:
    '''
    # if num_views == 6:
    #     cameras,base_dirs = init_6_canical_cams(res, device)
    #     return cameras, base_dirs
    assert distribution in ['fibonacci_sphere','self_defined','blender','exact_blender']
    if distribution == 'fibonacci_sphere':
        eye_positions = fibonacci_sphere(num_views, distance)
    elif distribution == 'blender' or distribution == 'exact_blender':
        num_views = 20
        phi = (1 + math.sqrt(5)) / 2.  # golden_ratio
        circumradius = math.sqrt(3)
        distance = circumradius * 1.2
        dodecahedron = [[-1, -1, -1],
                        [1, -1, -1],
                        [1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1],
                        [1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, 1],
                        [0, -phi, -1 / phi],
                        [0, -phi, 1 / phi],
                        [0, phi, -1 / phi],
                        [0, phi, 1 / phi],
                        [-1 / phi, 0, -phi],
                        [-1 / phi, 0, phi],
                        [1 / phi, 0, -phi],
                        [1 / phi, 0, phi],
                        [-phi, -1 / phi, 0],
                        [-phi, 1 / phi, 0],
                        [phi, -1 / phi, 0],
                        [phi, 1 / phi, 0]]

        eye_positions = np.array(dodecahedron).astype(float) * 1.2
        M = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0.0]
        ])
        eye_positions = eye_positions.dot(M.T)
    elif distribution == 'self_defined':
        if num_views ==6:
            eye_positions = distance * np.array([
                [0, 0, -1.0],
                [0, 0, 1.0],
                [0, -1.0, 0],
                [0, 1.0, 0],
                [-1.0, 0, 0],
                [1.0,0,0],
            ])
        elif num_views == 20:
            phi = (1 + math.sqrt(5)) / 2.  # golden_ratio
            circumradius = math.sqrt(3)
            distance = circumradius * 1.2
            dodecahedron = [[-1, -1, -1],
                            [1, -1, -1],
                            [1, 1, -1],
                            [-1, 1, -1],
                            [-1, -1, 1],
                            [1, -1, 1],
                            [1, 1, 1],
                            [-1, 1, 1],
                            [0, -phi, -1 / phi],
                            [0, -phi, 1 / phi],
                            [0, phi, -1 / phi],
                            [0, phi, 1 / phi],
                            [-1 / phi, 0, -phi],
                            [-1 / phi, 0, phi],
                            [1 / phi, 0, -phi],
                            [1 / phi, 0, phi],
                            [-phi, -1 / phi, 0],
                            [-phi, 1 / phi, 0],
                            [phi, -1 / phi, 0],
                            [phi, 1 / phi, 0]]

            eye_positions = np.array(dodecahedron).astype(float) #* 1.2

        # elif num_views == 8:
        #     eye_positions = distance * np.array([
        #             [-1, -1, -1],
        #             [1, -1, -1],
        #             [1, 1, -1],
        #             [-1, 1, -1],
        #             [-1, -1, 1],
        #             [1, -1, 1],
        #             [1, 1, 1],
        #             [-1, 1, 1],
        #         ])
  

    cameras = []
    base_dirs = torch.zeros((num_views,3),dtype=torch.float,device = device)
    up_dirs = torch.zeros((num_views,3),dtype=torch.float,device = device)
    fovy_angle = math.pi * 45 / 180
    if distribution=='exact_blender':
        fovy_angle = 0.8575560450553894
    for i,eye in enumerate(eye_positions):
        eye = np.array(eye)
        at = np.array([0,0,0])  # origin
        up = calculate_up_vector(eye, at)

        camera = kal.render.camera.Camera.from_args(eye=eye,
                                           at=at,
                                           up=up,
                                           # fov=math.pi * 45 / 180,
                                           fov=fovy_angle,
                                           width=res, height=res, device=device)

        cameras.append(camera)
        base_dir = eye-at
        base_dirs[i] = torch.tensor(base_dir).float().to(device)
        up_dirs[i] = torch.tensor(up).float().to(device)
    if vis:
        # eye_positions = np.array(eye_positions)
        base_dirs = base_dirs.detach().cpu().numpy()
        from vtk_basic import vis_actors_vtk, get_colorful_pc_actor_vtk,get_one_arrow_actor
        # vis_actors_vtk([get_pc_actor_vtk(pc_np=eye_positions,color=(1,0,0),point_size=10)],arrows=True)
        actors = [get_colorful_pc_actor_vtk(pc_np=eye_positions, point_size=12, opacity=1)]
        for i in range(num_views):
            actors.append(get_one_arrow_actor(center=eye_positions[i],vector=at-eye_positions[i]))
        vis_actors_vtk(actors, arrows=True)
    return cameras,base_dirs,eye_positions,up_dirs


# ------------------------------- render ----------------------------------
def render_textured_mesh2(vertices,faces,uvs,face_uvs_idx,atlas_img,
                          cams, rescale=True,uv_centers=0,uv_scales=2,padding=0,inpaint_scale_factors=None,
                          glctx=None,save_path=None,save=False ,normalize_mesh=False,render_height=None,render_width=None):

    if render_height is None:
        render_height = cams[0].height
        render_width = cams[0].width

    device = vertices.device
    # Here we are preprocessing the materials, assigning faces to materials and
    # using single diffuse color as backup when map doesn't exist (and face_uvs_idx == -1)
    uvs = torch.nn.functional.pad(uvs.unsqueeze(0).to(device), (0, 0, 0, 1)) #% 1. # don't %1 here. %1 after interpolate

    # print('uvs.shape',uvs.shape) # [1,?,2]

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0

    materials = [atlas_img.permute(2,0,1).unsqueeze(0)]

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = uvs.shape[1] - 1

    # normalize mesh
    if normalize_mesh:
        vertices_min = vertices.min(0)[0]
        vertices_max = vertices.max(0)[0]
        vertices -= (vertices_max + vertices_min) / 2.
        vertices /= (vertices_max - vertices_min).max()

    # render

    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=device)
    for i, cam in enumerate(cams):
        transformed_vertices = cam.transform(vertices.unsqueeze(0))
        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()

    # rescale if needed
    if rescale:
        vertice_uvs = pos[:, :, :2]
        vertice_uvs = (vertice_uvs - uv_centers) / uv_scales  # now all between -0.5, 0.5
        vertice_uvs = vertice_uvs * (1 - 2 * padding)  # now all between -0.45, 0.45
        vertice_uvs = vertice_uvs * inpaint_scale_factors.unsqueeze(-1).unsqueeze(-1)
        vertice_uvs = vertice_uvs + 0.5  # now all between 0.05, 0.95
        vertice_uvs = vertice_uvs.clip(0, 1)
        pos[:, :, :2] = vertice_uvs * 2 - 1  # use the rescaled result to calculate masks, faceids and depths

    rast = dr.rasterize(glctx, pos, faces.int(), resolution=[render_height, render_width],
                        grad_db=False)  # tuple of two tensors where we only use the first
    hard_mask = rast[0][:, :, :, -1:] != 0 # # cam_num,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous() # cam_num,res,res

    uv_map = nvdiffrast.torch.interpolate(uvs, rast[0], face_uvs_idx.int())[0] # cam_num,res,res,2, right here
    # print('uv_map.shape',uv_map.shape) # right here
    # print('uv_map.min(',uv_map.min(),uv_map.max())
    res = render_height
    # print('uvs.shape',uvs.shape) # 1,num_uvs,2
    # print('face_uvs_idx.shape',face_uvs_idx.shape) # face_num,3
    # print('uv_map.shape',uv_map.shape) # cam_num, res,res,2

    imgs = torch.zeros((len(cams), res, res, 3), dtype=torch.float, device=device).requires_grad_()
   
    # Obj meshes can be composed of multiple materials
    # so at rendering we need to interpolate from corresponding materials
    # im_material_idx = torch.zeros_like(face_idx).to(device)
    # im_material_idx[face_idx == -1] = -1 # cam_num, res,res
    im_material_idx = hard_mask.long()-1 # # cam_num,res,res,1
    im_material_idx = im_material_idx.squeeze(-1) # # cam_num,res,res

   

    # the following is relatively slow

    # each camera has an image
    # im_material_idx # [cam_num, res,res], material_idx of each pixel of each image
    # imgs # [cam_num, res, res, 3],
    # materials: list of material images, each element is of size [1,3,H,W]. Materials is with different H and W.
    # uv_map # [cam_num,res,res,2], uv coordiantes of each pixel of each image


    for cam_i in range(len(cams)):
        for i, material in enumerate(materials):
            # mask = im_material_idx[cam_i] == i # res,res # only foreground pixels
            mask = torch.ones_like(im_material_idx[cam_i]).bool()  # res,res # all pixels
            _texcoords = (uv_map[cam_i].unsqueeze(0) %1)* 2. - 1.  # cam_num,res,res,2 # %1->(0,1); *2-1 -> (-1,1)
       
            _texcoords[:, 1] = -_texcoords[:, 1]  # cam_num,res,res,2
       
            pixel_val = torch.nn.functional.grid_sample(
                materials[i], _texcoords.reshape(1, 1, -1, 2),
                mode='bilinear', align_corners=False,
                padding_mode='border')

      
            pixel_val.requires_grad_()
        

            # print('pixel_val',pixel_val.shape) # [1,3,1, num]
            # print('pixel_val[0, :, 0].permute(1, 0)',pixel_val[0, :, 0].permute(1, 0).shape) # [res*res,3]
            # print('imgs[cam_i]',imgs[cam_i].shape) #[res,res,3]
            imgs[cam_i][mask] = pixel_val[0, :, 0].permute(1, 0)#[0]
  

    # save
    if save:
        os.makedirs(save_path, exist_ok=True)
        save_with_alpha = True
        # hard_mask  # # cam_num,res,res,1
        for i, cam in enumerate(cams):
            img = imgs.cpu().numpy()[i, ::-1, :, :]  # Flip vertically. # H,W,3
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8

            if not save_with_alpha:
                imageio.imsave(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))), img) # without alpha channel

            else:
                alpha = hard_mask.cpu().numpy()[i, ::-1, :, :]  # H,W,1
                alpha = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8
                img = np.concatenate((img,alpha),axis=2) # H,W,4
                img = Image.fromarray(img, "RGBA")
                img.save(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))))


    imgs_flip = torch.flip(imgs,dims=[1]).requires_grad_() # Flip vertically. # N,H,W,3

    return imgs_flip.permute(0,3,1,2)

def render_textured_mesh(mesh_file,cams, device, save_path,
                         glctx,save=True, vertices = None,normalize_mesh=True,
                         light_dirs=None,gamma =None,double_side=False,
                         geo_only=False,color=[0.5,0.5,0.5]):
    # load mesh
    mesh = kal.io.obj.import_mesh(mesh_file, with_materials=True)

    if vertices is None:
        vertices = mesh.vertices.to(device)
    else:
        assert vertices.shape[0] == mesh.vertices.shape[0]
        # mesh.vertices = vertices # cannot set attributes
    faces = mesh.faces.to(device)
    face_normals = kal.ops.mesh.face_normals(face_vertices=mesh.vertices[mesh.faces].unsqueeze(0),
                                             unit=True)[0].to(device)  # [F,3]

    # Here we are preprocessing the materials, assigning faces to materials and
    # using single diffuse color as backup when map doesn't exist (and face_uvs_idx == -1)
    uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0).to(device), (0, 0, 0, 1)) #% 1. # don't %1 here. %1 after interpolate
    face_uvs_idx = mesh.face_uvs_idx.to(device)
    # face_material_idx = mesh.material_assignments.to(device) # my kaolin version is too low, use the following instead
    materials_order = mesh.materials_order
    nb_faces = faces.shape[0]

    num_consecutive_materials = \
        torch.cat([
            materials_order[1:, 1],
            torch.LongTensor([nb_faces])
        ], dim=0) - materials_order[:, 1]


    face_material_idx = kal.ops.batch.tile_to_packed(
        materials_order[:, 0],
        num_consecutive_materials
    ).squeeze(-1).to(device)

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0

    if geo_only:
        materials = [torch.tensor(np.array(color).reshape(1,3,1,1)).float().to(device) for m in mesh.materials]
    else:
        materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).to(device).float() / 255. if 'map_Kd' in m else
                     m['Kd'].reshape(1, 3, 1, 1).to(device)
                     for m in mesh.materials]

    nb_faces = faces.shape[0]

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = uvs.shape[1] - 1


    # normalize mesh
    if normalize_mesh:
        vertices_min = vertices.min(0)[0]
        vertices_max = vertices.max(0)[0]
        vertices -= (vertices_max + vertices_min) / 2.
        vertices /= (vertices_max - vertices_min).max()

    # render
    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=device)
    for i, cam in enumerate(cams):
        transformed_vertices = cam.transform(vertices.unsqueeze(0))
        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()
    rast = dr.rasterize(glctx, pos, faces.int(), resolution=[cam.height, cam.width],
                        grad_db=False)  # tuple of two tensors where we only use the first
    hard_mask = rast[0][:, :, :, -1:] != 0 # # cam_num,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous() # cam_num,res,res

    uv_map = nvdiffrast.torch.interpolate(uvs, rast[0], face_uvs_idx.int())[0] # cam_num,res,res,2

    res = cam.height
    # print('uvs.shape',uvs.shape) # 1,num_uvs,2
    # print('face_uvs_idx.shape',face_uvs_idx.shape) # face_num,3
    # print('uv_map.shape',uv_map.shape) # cam_num, res,res,2

    imgs = torch.zeros((len(cams), res, res, 3), dtype=torch.float, device=device)

    albedos = torch.zeros((len(cams), res, res, 3), dtype=torch.float, device=device)
    # Obj meshes can be composed of multiple materials
    # so at rendering we need to interpolate from corresponding materials
    im_material_idx = face_material_idx[face_idx]
    im_material_idx[face_idx == -1] = -1 # cam_num, res,res



    for cam_i in range(len(cams)):

        for i, material in enumerate(materials):
            mask = im_material_idx[cam_i].unsqueeze(0) == i #
            mask_idx = torch.nonzero(mask, as_tuple=False)
            _texcoords = (uv_map[cam_i].unsqueeze(0)[mask] %1)* 2. - 1.  # cam_num,res,res,2 # %1->(0,1); *2-1 -> (-1,1)
            # print('_texcoords.shape',_texcoords)
            _texcoords[:, 1] = -_texcoords[:, 1]  # cam_num,res,res,2
            pixel_val = torch.nn.functional.grid_sample(
                materials[i], _texcoords.reshape(1, 1, -1, 2),
                mode='bilinear', align_corners=False,
                padding_mode='border')
            albedos[cam_i].unsqueeze(0)[mask] = pixel_val[0, :, 0].permute(1, 0)

        '''
        light_dirs = torch.tensor(np.array([
        [0.8, 0, 0],  # left
        [0.0, 0.5, 0.0],  # top
        [0.0, 0.0, -0.5],  # front
        ])).float().to(device)
        '''
        if light_dirs is not None:
            face_idx_valid = face_idx.clone()
            face_idx_valid[face_idx < 0] = 0

            # fix normal to point outwards
            camera_forward = cams[cam_i].extrinsics.R[0][2]  # [3] # R is [1,3,3]
            # print('camera_forward',camera_forward) # [0.5774,0.5774,-0.5774]
            temp = torch.matmul(face_normals, camera_forward.unsqueeze(1))  # F
            face_normals[temp.squeeze(-1)<0] *=-1
            #
            view_pixel_normals = face_normals[face_idx_valid[cam_i]]  # res,res,3


            for light_dir in light_dirs:
                # here
                VN_dot_Light_dir = torch.matmul(view_pixel_normals.reshape(-1, 3), light_dir.unsqueeze(1))
                VN_dot_Light_dir = VN_dot_Light_dir.reshape(res, res, 1)  # .squeeze(-1)
                vis = False
                if vis:
                    cat = cat_images(albedos[cam_i].permute(2, 0, 1).detach().cpu().numpy(),
                                     VN_dot_Light_dir.repeat(1, 1, 3).permute(2, 0, 1).detach().cpu().numpy())

                    cat = cat_images(cat, (albedos[cam_i] * VN_dot_Light_dir).permute(2, 0, 1).detach().cpu().numpy())
                    # cat = cat_images(cat, hard_mask[cam_i].repeat(1, 1, 3).permute(2, 0, 1).detach().cpu().numpy())

                    cat2 = cat_images(albedos[cam_i].permute(2, 0, 1).detach().cpu().numpy(),
                                     torch.abs(VN_dot_Light_dir).repeat(1, 1, 3).permute(2, 0, 1).detach().cpu().numpy())
                    cat2 = cat_images(cat2, (albedos[cam_i] * torch.abs(VN_dot_Light_dir)).permute(2, 0,
                                                                                              1).detach().cpu().numpy())
                    cat = cat_images(cat,cat2,horizon=False)
                    display_CHW_RGB_img_np_matplotlib(cat)
                if double_side:
                    VN_dot_Light_dir = torch.abs(VN_dot_Light_dir) # double sided
                # imgs[cam_i] += albedos[cam_i] * VN_dot_Light_dir
                imgs[cam_i] += albedos[cam_i] * VN_dot_Light_dir.clip(0, 1)

            imgs[cam_i][~hard_mask[cam_i].squeeze(-1)] = 0

            imgs[cam_i] = imgs[cam_i].clip(0, 1)
            if gamma is not None:
                imgs[cam_i] = imgs[cam_i] ** (1.0 / gamma)  # gamma
        else:
            imgs[cam_i] = albedos[cam_i]

    # save
    if save:
        os.makedirs(save_path, exist_ok=True)
        save_with_alpha = True
        # hard_mask  # # cam_num,res,res,1
        for i, cam in enumerate(cams):
            img = imgs.cpu().numpy()[i, ::-1, :, :]  # Flip vertically. # H,W,3
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8

            if not save_with_alpha:
                imageio.imsave(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))), img) # without alpha channel

            else:
                alpha = hard_mask.cpu().numpy()[i, ::-1, :, :]  # H,W,1
                alpha = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8
                img = np.concatenate((img,alpha),axis=2) # H,W,4
                img = Image.fromarray(img, "RGBA")
                img.save(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))))


    imgs = torch.flip(imgs,dims=[1]) # Flip vertically. # N,H,W,3
    return imgs.permute(0,3,1,2) # N,3,H,W

def render_textured_mesh_w_mask(mesh_file,cams, device, save_path,glctx,save=True, vertices = None,normalize_mesh=True):
    # load mesh
    mesh = kal.io.obj.import_mesh(mesh_file, with_materials=True)
 
    if vertices is None:
        vertices = mesh.vertices.to(device)
    else:
        assert vertices.shape[0] == mesh.vertices.shape[0]
        # mesh.vertices = vertices # cannot set attributes
    faces = mesh.faces.to(device)
    # Here we are preprocessing the materials, assigning faces to materials and
    # using single diffuse color as backup when map doesn't exist (and face_uvs_idx == -1)
    uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0).to(device), (0, 0, 0, 1)) #% 1. # don't %1 here. %1 after interpolate
    face_uvs_idx = mesh.face_uvs_idx.to(device)
    # face_material_idx = mesh.material_assignments.to(device) # my kaolin version is too low, use the following instead
    materials_order = mesh.materials_order
    nb_faces = faces.shape[0]

    num_consecutive_materials = \
        torch.cat([
            materials_order[1:, 1],
            torch.LongTensor([nb_faces])
        ], dim=0) - materials_order[:, 1]


    face_material_idx = kal.ops.batch.tile_to_packed(
        materials_order[:, 0],
        num_consecutive_materials
    ).squeeze(-1).to(device)

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0

    materials = [m['map_Kd'].permute(2, 0, 1).unsqueeze(0).to(device).float() / 255. if 'map_Kd' in m else
                 m['Kd'].reshape(1, 3, 1, 1).to(device)
                 for m in mesh.materials]
    nb_faces = faces.shape[0]

    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = uvs.shape[1] - 1


    # normalize mesh
    if normalize_mesh:
        vertices_min = vertices.min(0)[0]
        vertices_max = vertices.max(0)[0]
        vertices -= (vertices_max + vertices_min) / 2.
        vertices /= (vertices_max - vertices_min).max()

    # render
    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=device)
    for i, cam in enumerate(cams):
        transformed_vertices = cam.transform(vertices.unsqueeze(0))
        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()
    rast = dr.rasterize(glctx, pos, faces.int(), resolution=[cam.height, cam.width],
                        grad_db=False)  # tuple of two tensors where we only use the first
    hard_mask = rast[0][:, :, :, -1:] != 0 # # cam_num,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous() # cam_num,res,res

    uv_map = nvdiffrast.torch.interpolate(uvs, rast[0], face_uvs_idx.int())[0] # cam_num,res,res,2

    res = cam.height
    # print('uvs.shape',uvs.shape) # 1,num_uvs,2
    # print('face_uvs_idx.shape',face_uvs_idx.shape) # face_num,3
    # print('uv_map.shape',uv_map.shape) # cam_num, res,res,2

    imgs = torch.zeros((len(cams), res, res, 3), dtype=torch.float, device=device)
    # Obj meshes can be composed of multiple materials
    # so at rendering we need to interpolate from corresponding materials
    im_material_idx = face_material_idx[face_idx]
    im_material_idx[face_idx == -1] = -1 # cam_num, res,res


    # the following is relatively slow

    # each camera has an image
    # im_material_idx # [cam_num, res,res], material_idx of each pixel of each image
    # imgs # [cam_num, res, res, 3],
    # materials: list of material images, each element is of size [1,3,H,W]. Materials is with different H and W.
    # uv_map # [cam_num,res,res,2], uv coordiantes of each pixel of each image

    for cam_i in range(len(cams)):
        for i, material in enumerate(materials):
            mask = im_material_idx[cam_i].unsqueeze(0) == i #
            mask_idx = torch.nonzero(mask, as_tuple=False)
            _texcoords = (uv_map[cam_i].unsqueeze(0)[mask] %1)* 2. - 1.  # cam_num,res,res,2 # %1->(0,1); *2-1 -> (-1,1)
      
            _texcoords[:, 1] = -_texcoords[:, 1]  # cam_num,res,res,2
            pixel_val = torch.nn.functional.grid_sample(
                materials[i], _texcoords.reshape(1, 1, -1, 2),
                mode='bilinear', align_corners=False,
                padding_mode='border')
            imgs[cam_i].unsqueeze(0)[mask] = pixel_val[0, :, 0].permute(1, 0)

    # save
    if save:
        os.makedirs(save_path, exist_ok=True)
        save_with_alpha = True
        # hard_mask  # # cam_num,res,res,1
        for i, cam in enumerate(cams):
            img = imgs.cpu().numpy()[i, ::-1, :, :]  # Flip vertically. # H,W,3
            img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8

            if not save_with_alpha:
                imageio.imsave(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))), img) # without alpha channel

            else:
                alpha = hard_mask.cpu().numpy()[i, ::-1, :, :]  # H,W,1
                alpha = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8
                img = np.concatenate((img,alpha),axis=2) # H,W,4
                img = Image.fromarray(img, "RGBA")
                img.save(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))))




    imgs = torch.flip(imgs,dims=[1]) # Flip vertically. # N,H,W,3
    return imgs.permute(0,3,1,2)



def render_textured_meshes_shapenet2(names=None, root_path=None, device=None,
                                     save_root_path=None,glctx=None):
    now = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M.%S')
    logger = get_logger(os.path.join(root_path,f'{now}_render_mesh.log'))

    if glctx is None:
        glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device)
    cameras, base_dirs, eye_positions,_ = create_cameras(num_views=20, distance=1.6, res=1024,device=device,distribution='self_defined')
    cls_ids = os.listdir(os.path.join(root_path, 'meshes'))
    for cls_id in cls_ids:
        # if cls_id != '02958343': # debug
        #     continue
        if cls_id.endswith('.log') or cls_id.endswith('.yaml') or cls_id.endswith('.py'):
            continue
        # if names is None:
        if True:
            names = os.listdir(os.path.join(root_path,'meshes',cls_id))
   
        for i,name in enumerate(names):
            # if i < 287: # debug
            #     continue
            print(cls_id, i, '/',len(names), name)

            mesh_file = os.path.join(root_path,'meshes', cls_id, name, 'models', 'model_normalized.obj')
            if not os.path.exists(mesh_file):
                mesh_file = os.path.join(root_path, 'meshes', cls_id, name, 'models', f'{name}.obj')
            if not os.path.exists(mesh_file):
                mesh_file = os.path.join(root_path, 'meshes', cls_id, name, 'meshes', 'model.obj') # google scanned objects
            if not os.path.exists(mesh_file):
                mesh_file = os.path.join(root_path, 'meshes', cls_id, name, 'Scan', 'Scan.obj') # omniobject
            save_path = os.path.join(root_path, 'rendered_imgs', cls_id, name)

            # skip exist
            if os.path.exists(save_path):
                if len(os.listdir(save_path)) ==20:
                    # print('skip exist',save_path)
                    continue
                else:
                    print('not skipt exist,', save_path)
            try:
                render_textured_mesh(mesh_file, cameras, device, save_path,save=True,glctx=glctx)
            except KeyboardInterrupt:
                # traceback.format_exc()
                # print((traceback.format_exc()))

                # logger.error((traceback.format_exc()))
                logger.error('key board interrupt')
                sys.exit(0)
            except:
                logger.error((f'{i},{name},{mesh_file}'))
                logger.error((traceback.format_exc()))




def render_per_vertex_color_mesh(mesh_file,cams,device,save_path,glctx,
                                 light_dirs=None,gamma=None,double_side=False):
    # load mesh
    import open3d as o3d
    # print('mesh_file',mesh_file)
    assert os.path.exists(mesh_file)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors)

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    col = torch.tensor(vertex_colors, dtype=torch.float32, device=device)
    tri = torch.tensor(triangles, dtype=torch.int32, device=device)

    face_normals = kal.ops.mesh.face_normals(face_vertices=vertices[triangles].unsqueeze(0),
                                             unit=True)[0].to(device)  # [F,3]

    # render
    res = cams[0].height

    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=device)
    for i, cam in enumerate(cams):
        transformed_vertices = cam.transform(vertices.unsqueeze(0))
        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()
        # pos must be [batch_num, point_num,4], produced by cam.transform and pad

    rast = dr.rasterize(glctx, pos, tri,resolution=[cam.height, cam.width], grad_db=False)  # tuple of two tensors where we only use the first

    hard_mask = rast[0][:, :, :, -1:] != 0  # # cam_num,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous()  # cam_num,res,res

    out = dr.interpolate(col, rast[0], tri)  # tuple of two tensors where we only use the first
    out = out[0]

    # save
    save_with_alpha = True
    os.makedirs(save_path,exist_ok=True)
    for i, cam in enumerate(cams):
        # albedo = out[i, ::-1, :, :]  # Flip vertically. (np)
        albedo = out[i]

        img =torch.zeros_like(albedo).to(device)
        if light_dirs is not None:
            face_idx_valid = face_idx.clone()
            face_idx_valid[face_idx < 0] = 0

            # fix normal to point outwards
            camera_forward = cams[i].extrinsics.R[0][2]  # [3] # R is [1,3,3]
          
            temp = torch.matmul(face_normals, camera_forward.unsqueeze(1))  # F
            face_normals[temp.squeeze(-1)<0] *=-1

            view_pixel_normals = face_normals[face_idx_valid[i]]  # res,res,3

            for light_dir in light_dirs:
                # here
                VN_dot_Light_dir = torch.matmul(view_pixel_normals.reshape(-1, 3), light_dir.unsqueeze(1))
                if double_side:
                    VN_dot_Light_dir = torch.abs(VN_dot_Light_dir)  # double sided
                VN_dot_Light_dir = VN_dot_Light_dir.reshape(res, res, 1)  # .squeeze(-1)

                # img += albedo * VN_dot_Light_dir
                img += albedo * VN_dot_Light_dir.clip(0, 1)
                # vis = True
                # if vis:
                #     cat = cat_images(albedo.permute(2, 0, 1).detach().cpu().numpy(),
                #                      VN_dot_Light_dir.repeat(1, 1, 3).permute(2, 0, 1).detach().cpu().numpy())
                #     cat = cat_images(cat, hard_mask[i].repeat(1, 1, 3).permute(2, 0, 1).detach().cpu().numpy())
                #     display_CHW_RGB_img_np_matplotlib(cat)
            img[~hard_mask[i].squeeze(-1)] = 0

            img = img.clip(0, 1)
            if gamma is not None:
                img = img ** (1.0 / gamma)  # gamma
        else:
            img = albedo
        img = img.cpu().numpy()
        img = img[ ::-1, :, :]  # Flip vertically. (np)
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8


        if not save_with_alpha:
            imageio.imsave(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))), img) # without alpha channel

        else:
            alpha = hard_mask.cpu().numpy()[i, ::-1, :, :]  # H,W,1
            alpha = np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8)  # Quantize to np.uint8
            img = np.concatenate((img,alpha),axis=2) # H,W,4
            img = Image.fromarray(img, "RGBA")
            img.save(os.path.join(save_path, 'albedo_{:s}.png'.format(str(i + 1).zfill(3))))


def render_per_vertex_color_meshes_shapenet2(mesh_root_path,device,glctx=None):
    if glctx is None:
        try:
            glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device)  #
        except:
            glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
    cameras, base_dirs, eye_positions,_ = create_cameras(num_views=20, distance=1.6, res=1024,device=device,distribution='self_defined')
    cls_ids = os.listdir(os.path.join(mesh_root_path, 'meshes'))
    for cls_id in cls_ids:
        if cls_id.endswith('.log'):
            continue
        if cls_id.endswith('.py'):
            continue
        names = os.listdir(os.path.join(mesh_root_path, 'meshes',cls_id))
        for i,name in enumerate(names):
            # if i < 603: # debug
            #     continue
            if name.endswith('.log'):
                continue
            if name.endswith('.py'):
                continue
            print(cls_id, i, '/',len(names), name)
            mesh_file = os.path.join(mesh_root_path,'meshes',cls_id,name,'models','model_normalized.obj')
            save_path = os.path.join(mesh_root_path,'rendered_imgs',cls_id,name)
            render_per_vertex_color_mesh(mesh_file, cameras, device, save_path,glctx)

def render_normal_map(vertices,faces,cams,glctx):
    device = vertices.device
    res=cams[0].height

    # get face normals
    face_normals = kal.ops.mesh.face_normals(face_vertices=vertices[faces].unsqueeze(0),
                                             unit=True)  # [1,num_faces,3) # all elements are between -1 to 1

    face_normals = face_normals[0] # [num_faces,3]
    face_normals = (face_normals+1)/2 # now all elements are between 0 to 1
    # print('face_normals',face_normals.min(),face_normals.max())

    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=device)
    vertex_depths = -1.0 * torch.ones((len(cams), vertices.shape[0]), device=device)
    for i, cam in enumerate(cams):
        transformed_vertices = cam.transform(vertices.unsqueeze(0))

        vertex_depths[i] = cam.extrinsics.transform(vertices.unsqueeze(0))[0,:,2] # from [1,N,3] to [N], somehow negative
        vertex_depths[i] = -vertex_depths[i] # now the depth values are positive
        vertex_depths[i] = (vertex_depths[i] -1.0)/(3.0-1.0) # normalize

        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()
  
    rast = dr.rasterize(glctx, pos, faces.int(), resolution=[cam.height, cam.width],
                        grad_db=False)  # tuple of two tensors where we only use the first
    hard_mask = rast[0][:, :, :, -1:] != 0  # # cam_num,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous()  # cam_num,res,res

    # flatten normals then reshape back
    face_idx[face_idx<0] = len(face_normals)
    face_normals_expanded = torch.cat([face_normals,torch.zeros((1,3),device=device)],0)
    normal_maps_flatten = face_normals_expanded[face_idx.flatten()] # [cam_num*res*res,3]
    normal_maps = normal_maps_flatten.view(len(cams),res,res,3)
    normal_maps = normal_maps.permute(0,3,1,2) # cam_num,3,res,res

    # get depth map # don't forget to unsqueeze it or it will be wrong
    depth_maps = dr.interpolate(vertex_depths.unsqueeze(-1), rast[0], faces.int())[0] # [cam_num,res,res,1]
    depth_maps = depth_maps.permute(0,3,1,2) # # [cam_num,1,res,res]
    depth_maps = depth_maps.repeat(1,3,1,1) # [cam_num,3,res,res]
  
    return normal_maps,depth_maps

def vis_cameras():
    # distribution in ['fibonacci_sphere', 'azimuth', 'self_defined']

    # create_cameras(num_views=6, distribution='self_defined', vis=True)
    # create_cameras(num_views=20, distribution='self_defined', vis=True)
    # create_cameras(num_views=20, distribution='fibonacci_sphere', vis=True)
    create_cameras(num_views=6, distribution='fibonacci_sphere', vis=True)



# ----------------------------------- trans
def camera_location_from_azimuth(azimuth, elevation, dist):
    """get camera_location (x, y, z)

    you can write your own version of camera_location function
    to return the camera loation in the blender world coordinates
    system

    Args:
        azimuth: azimuth radius(object centered)
        elevation: elevation radius(object centered)
        dist: distance between camera and object(in meter)

    Returens:
        return the camera location in world coordinates in meters
    """

    phi = float(elevation)
    theta = float(azimuth)
    dist = float(dist)

    x = dist * math.cos(phi) * math.cos(theta)
    y = dist * math.cos(phi) * math.sin(theta)
    z = dist * math.sin(phi)

    return x, y, z


def get_cam_Ks_RTs_from_locations(cam_locations):
    '''
    The returned cam_RTs are used to transform from world 2 cam coordinate system.
    # https://zhuanlan.zhihu.com/p/561394626
    target is always at [0,0,0]
    :param cam_locations: [num_cam,3]
    :return: cam_Ks (num_cam x 3 x 3), cam_RTs (num_cam x 3 x 4)
    '''
    num_cam = len(cam_locations)
    # cam_Ks = np.zeros((num_cam, 3, 3))
    cam_RTs = np.zeros((num_cam, 3, 4))
    target = np.array([0,0,0])
    default_up = np.array([0,1,0])
    backup_up = np.array([0,0,1])
    for i in range(num_cam):
        eye = cam_locations[i]

        N = target - eye # look direction
        N = N/np.linalg.norm(N)
        if N[0] ==0 and N[2] ==0: # if N and up overlap
            up = backup_up
        else:
            up = default_up
        U = np.cross(N,up)
        U = U/np.linalg.norm(U)

        V = np.cross(U,N)
        V = V/np.linalg.norm(V)
        RT = np.array([ # world 2 camera
            [U[0], U[1], U[2], np.dot(-U,eye)],
            [V[0], V[1], V[2], np.dot(-V,eye)],
            [N[0], N[1], N[2], np.dot(-N,eye)],
            # [0, 0, 0, 1]
        ])
        cam_RTs[i] = RT
    cam_K = np.array([
        [560.0, 0, 256,],
        [0,560, 256,],
        [0, 0, 1,],
    ])

    cam_Ks = np.repeat(cam_K[np.newaxis,...],len(cam_locations),axis=0)
    # cam_RTs = cam_RTs.astype(np.float32)
    # cam_Ks = cam_Ks.astype(np.float32)
    return cam_Ks, cam_RTs


if __name__ == '__main__':
    '''vis the cameras used for rendering multi-view images'''
    # vis_cameras()

    '''render SPR'''
    # root_path = 'out_inference/2023.07.20.15.02.58_SPR_noise_None_kaolin_GT_normal'
    # save_root_path = root_path.replace('out_inference','nvdiffrast_rendered_images')
    # render_per_vertex_color_meshes_shapenet2(root_path=root_path, device=device, save_root_path=save_root_path)

    '''render GT'''
    mesh_root_path = '/home/me/RfDNet/datasets/ShapeNetCore.v2.clean_mesh'
    save_root_path = 'nvdiffrast_rendered_images/GT_self_defined_20'
    render_textured_meshes_shapenet2(root_path=mesh_root_path, device=device, save_root_path=save_root_path)

    '''render ours'''
    # root_path = 'out_inference/2023.07.19.11.30.46_TextureField_noise_0.005_kaolin'
    # save_root_path = root_path.replace('out_inference','nvdiffrast_rendered_images')
    # render_textured_meshes_shapenet2(root_path=root_path, device=device, save_root_path=save_root_path)

    '''test render with lighting'''
    # https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/diffuse_lighting.ipynb
