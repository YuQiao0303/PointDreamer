import sys

from scipy.interpolate import griddata
from torch.nn.functional import grid_sample, interpolate
import torch.nn.functional as F
# sys.path.append('..')
# sys.path.append('.')
import os

import numpy as np
import torch
import kaolin as kal
import nvdiffrast
from torchvision.transforms import transforms
import nvdiffrast.torch as dr
import trimesh
from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images,save_CHW_RGB_img,load_CHW_RGB_img

import open3d as o3d
import PIL
import torch.nn.functional as F

import traceback
import time

import cv2
from utils.camera_utils import render_textured_mesh,render_textured_meshes_shapenet2
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
# from data.load_data_like_shapenet_core_v2 import pc_collate_fn_np
from utils.other_utils import save_colored_pc_ply

from utils.utils_2d import display_CHW_RGB_img_np_matplotlib
from utils.utils_2d import detect_abnormal_bright_spots_in_gray_img
# device = torch.device('cuda')
# glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device)
from utils.camera_utils import get_cam_Ks_RTs_from_locations
from models.get3d.extract_texture_map import xatlas_uvmap_w_face_id
from models.get3d.get3d_utils.utils_3d import save_obj, savemeshtes2
from utils.camera_utils import render_textured_mesh2
from pointdreamer.unproject import get_shrinked_per_view_per_pixel_visibility_torch
# from pointdreamer.unproject import get_charts, get_background_edge, get_border_area_in_atlax_view_id
# from utils.utils_2d import detect_edges_in_gray_by_scharr_torch_batch,dilate_torch_batch

def project_poitns2mesh_by_sample_CD(points,mesh_verts,mesh_faces,sample_num):
    '''
    :param points: N,3
    :param mesh_verts:
    :param mesh_faces:
    :return: N,3
    '''
    mesh_points, mesh_points_face_ids = kal.ops.mesh.sample_points(vertices=mesh_verts.unsqueeze(0), faces=mesh_faces,
                                                                   num_samples=sample_num, areas=None,
                                                                   face_features=None)

    distance, idx = kal.metrics.pointcloud.sided_distance(p1=points.unsqueeze(0), p2=mesh_points) # 1,points_num
    idx = idx[0]
    projected_points = mesh_points[0][idx]
    projected_points_face_ids = mesh_points_face_ids[0][idx]
    return projected_points,projected_points_face_ids


# ----------------------- 3D to 2D --------------------
def point_xyz2pixel_uv(points,cam,res =256):
    assert cam.width == cam.height
    # res = cam.width
    transformed_vertices = cam.transform(points.unsqueeze(0)) # NDC space coordiantes: all between -1 and 1
    point_uvs = (transformed_vertices[0][:, :2] +1) *0.5
    point_depth = transformed_vertices[0][ :, 2]
    point_pixels = transformed_vertices[0][:,:2]*res/2 + res/2
    point_pixels = point_pixels.long()
    point_pixels = torch.cat((point_pixels[:,1].unsqueeze(1),point_pixels[:,0].unsqueeze(1)),dim=1) # switch x and y

    # clip to make sure they are right
    point_uvs = point_uvs.clip(0,1)
    point_pixels = point_pixels.clip(0,res-1)

    return point_pixels,point_uvs,point_depth


def get_rendered_hard_mask_and_face_idx(cam, vertices, faces,glctx):
    transformed_vertices = cam.transform(vertices.unsqueeze(0))

    # Create a fake W (See nvdiffrast documentation)
    pos = torch.nn.functional.pad(transformed_vertices, (0, 1), mode='constant', value=1.).contiguous()# 1,point_num,4
    rast = nvdiffrast.torch.rasterize(glctx, pos, faces.int(), (cam.height, cam.width), grad_db=False)
    hard_mask = rast[0][:, :, :, -1:] != 0 # 1,res,res,1
    face_idx = (rast[0][..., -1].long() - 1).contiguous() # 1,res,res
    normalized_depth = rast[0][..., 2]  # [1,res,res]

    return hard_mask[0],face_idx[0],normalized_depth[0],transformed_vertices[0] # delete batch

def get_rendered_hard_mask_and_face_idx_batch(cams, vertices, faces, points,glctx,rescale=True,padding = 0.05):
    pos = torch.zeros((len(cams), vertices.shape[0], 4), device=vertices.device)
    transformed_points = torch.zeros((len(cams), points.shape[0], 3), device=vertices.device)

    for i, cam in enumerate(cams):
        # transformed_vertices = cam.transform(vertices.unsqueeze(0))
        transformed_vertices_points = cam.transform(torch.cat([vertices,points],0)) # [all_point_num,3]

        transformed_vertices = transformed_vertices_points[:len(vertices),:]
        transformed_points[i] = transformed_vertices_points[len(vertices):,:]
        # Create a fake W (See nvdiffrast documentation)
        pos[i] = torch.nn.functional.pad(
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous()


    #####
    if rescale:
        vertice_uvs = pos[:, :, :2]
        ori_vertice_uvs_min = vertice_uvs.min(1)[0] # cam_num,2
        ori_vertice_uvs_max = vertice_uvs.max(1)[0] # cam_num,2

        ori_vertice_uvs_min = ori_vertice_uvs_min.unsqueeze(1) # cam_num,1,2
        ori_vertice_uvs_max = ori_vertice_uvs_max.unsqueeze(1) # cam_num,1,2
        uv_centers = (ori_vertice_uvs_min + ori_vertice_uvs_max) / 2  # cam_num,1,2
        uv_scales = (ori_vertice_uvs_max-ori_vertice_uvs_min).max(2)[0].unsqueeze(2)  # cam_num,1,2
        vertice_uvs = (vertice_uvs - uv_centers) / uv_scales # now all between -0.5, 0.5
        vertice_uvs = vertice_uvs * (1-2*padding) # now all between -0.45, 0.45
        vertice_uvs = vertice_uvs+0.5 # now all between 0.05, 0.95
        vertice_uvs = vertice_uvs.clip(0, 1)
        pos[:, :, :2] = vertice_uvs *2-1 # use the rescaled result to calculate masks, faceids and depths

        point_uvs = transformed_points[..., :2]
        point_uvs = (point_uvs - uv_centers) / uv_scales  # now all between -0.5, 0.5
        point_uvs = point_uvs * (1 - 2 * padding)  # now all between -0.45, 0.45
        point_uvs = point_uvs + 0.5  # now all between 0.05, 0.95

        point_depths = transformed_points[:, :, 2]  # # [num_cameras,point_num]
    else:
        vertice_uvs = (pos[:, :, :2] + 1) * 0.5  ## cam_num,vertice_num,2
        vertice_uvs = vertice_uvs.clip(0, 1)

        point_uvs = transformed_points[..., :2]
        point_uvs = (point_uvs +1) * 0.5  #

        uv_centers = 0
        uv_scales = 2
        padding = 0
        point_depths = transformed_points[:, :, 2]
    rast = nvdiffrast.torch.rasterize(glctx, pos, faces.int(), resolution=[cam.height, cam.width],
                        grad_db=False)  # tuple of two tensors where we only use the first
    hard_masks = rast[0][:, :, :, -1:] != 0  # # cam_num,res,res,1
    hard_masks = hard_masks.squeeze(-1)  # # cam_num,res,res
    face_idxs = (rast[0][..., -1].long() - 1).contiguous()  # cam_num,res,res
    mesh_normalized_depths = rast[0][..., 2]  # [num_cameras,res,res]


    return hard_masks,face_idxs,mesh_normalized_depths,vertice_uvs,uv_centers,uv_scales,padding,point_uvs,point_depths


def get_point_validation_by_depth(cam_res,point_uvs,point_depths,mesh_depths,offset = 0,vis=False):
    cam_num, point_num,_ = point_uvs.shape
    device = point_uvs.device
    point_visibility = torch.zeros((cam_num, point_num), device=point_uvs.device).bool()
    point_pixels = point_uvs * cam_res
    point_pixels = point_pixels.clip(0, cam_res - 1)
    point_pixels = point_pixels.long()
    point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                             dim=-1)  # switch x and y if you ever need to query pixel coordiantes

    reference_depth = mesh_depths[
        torch.arange(cam_num).view(-1, 1),  # cam_num, 1
        point_pixels[:, :, 0].long(),  # cam_num, point_num
        point_pixels[:, :, 1].long()  # cam_num, point_num
    ]

    if vis:
        for i in range(cam_num):
            # reference depth img
            depth_img = mesh_depths[i].unsqueeze(0).repeat(3, 1, 1)
            depth_img = depth_img
            mask = depth_img != 0
            depth_min, depth_max = depth_img[mask].min(), depth_img[mask].max()
            # print('depth_min', depth_min)
            # print('depth_max', depth_max)
            depth_img = (depth_img - depth_min) / (depth_max - depth_min)
            depth_img[~mask] = 0
            img1 = depth_img.clone()[0]
            img2 = depth_img.clone()[0]
            depth_img = depth_img.detach().cpu().numpy()
  

            img1[point_pixels[i, :, 0], point_pixels[i, :, 1]] = (point_depths[i] - depth_min) / (depth_max - depth_min)
            img1 = img1.unsqueeze(0).repeat(3, 1, 1)
            # img1 = (img1 - depth_min) / (depth_max - depth_min)
            img1[~mask] = 0
            img1 = img1.detach().cpu().numpy()


            img2 = img2.unsqueeze(0).repeat(3, 1, 1)
            img2[:,point_pixels[i, :, 0], point_pixels[i, :, 1]] = torch.tensor([1.0,0,0]).float().to(device).unsqueeze(1)

            img2 = img2.detach().cpu().numpy()

            cat = cat_images(img1,img2)
            display_CHW_RGB_img_np_matplotlib(cat)

    point_visibility[point_depths - reference_depth <= offset] = True  # # [num_cameras,point_num]

    return point_visibility,point_pixels.long()
    ###############################################
def get_point_validation_by_o3d(points,eye_positions = None, hidden_point_removal_radius=None,):
    point_visibility = torch.zeros((len(eye_positions), points.shape[0]), device=points.device).bool()

    for i_cam in range(len(eye_positions)):

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points.cpu().numpy()))

        # o3d_camera = [0, 0, diameter]
        o3d_camera = np.array(eye_positions[i_cam])

        _, pt_map = pcd.hidden_point_removal(o3d_camera, hidden_point_removal_radius)
      
        vis = False
        if vis:
            pcd = pcd.select_by_index(pt_map)
            geoms = [pcd]
            o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

        visible_point_ids = np.array(pt_map)

        point_visibility[i_cam, visible_point_ids] = True
    return point_visibility

def refine_point_validation(cam_RTs,cam_K,res,

                            hard_masks,point_validation,point_uvs,points,save_path):
    view_num = len(cam_RTs)

    point_pixels = point_uvs * res  # [num_cameras,piont_num,2]
    point_pixels = point_pixels.long()
    point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                             dim=-1)  # switch x and y
    point_pixels = point_pixels.clip(0, res - 1)

    sparse_depth_maps = -100.0 * torch.ones((view_num,res,res)).to(point_pixels.device)
    abnormal_masks = torch.zeros((view_num,res,res)).to(point_pixels.device)
    new_point_validation = point_validation.clone()

    for i in range(view_num):
        start = time.time()
       

        # 1. paint sparse depth map
        cam_RT = cam_RTs[i]

        # points[point_validation[i]]: [N,3];        cam_RT: [3,4]
        point_cam = torch.matmul(points[point_validation[i]],cam_RT[:, :-1].T) + cam_RT[:, -1]


        z = point_cam[:, 2]  # [N]
       

        sparse_depth_maps[i] = \
            paint_pixels(sparse_depth_maps[i].unsqueeze(0), point_pixels[i][point_validation[i]], z.unsqueeze(-1), point_size=1)[0]

        non_empty_pixel_mask = sparse_depth_maps[i]!= -100 #
        foreground_mask = transforms.Resize((res,res))(hard_masks[i].unsqueeze(0))[0]

        no_need_inpaint_mask = non_empty_pixel_mask

        # 2. naively fill dense depth
        dense_depth_map = naive_inpainting(img=sparse_depth_maps[i].unsqueeze(0), # [1,res,res]
                                               no_need_inpaint_mask2=no_need_inpaint_mask.unsqueeze(0), # [1,res,res]
                                           method='nearest')[0]


        min_for_norm,max_for_norm = 0.5,2.5

        sparse = sparse_depth_maps[i].detach().cpu().numpy() # [res,res]
        dense = dense_depth_map

        vis = False
        if vis:
            sparse_normed = (sparse - min_for_norm) / (max_for_norm - min_for_norm)
            sparse_normed[sparse_normed<0] = 0 #
            sparse_normed = np.repeat(sparse_normed[np.newaxis, ...], 3, axis=0).astype(np.float64) # [3,res,res]
            display_CHW_RGB_img_np_matplotlib(sparse_normed)


            dense_normed = (dense_normed - min_for_norm) / (max_for_norm - min_for_norm)
            dense_normed[dense_normed<0] = 0
            dense_normed = np.repeat(dense_normed[np.newaxis, ...], 3, axis=0).astype(np.float64)  # [3,res,res]
            display_CHW_RGB_img_np_matplotlib(dense_normed)

        # 3. get abnormal foreground_mask by rmoving abnormal bright spots
        abnormal_mask = detect_abnormal_bright_spots_in_gray_img(dense,
                                                                 foreground_mask=foreground_mask.cpu().numpy().astype(np.bool_),
                                                                 save_path=os.path.join(save_path, f'{i}_depth.png'),
                                                                 min_for_norm=min_for_norm, max_for_norm=max_for_norm,
                                                                 edge_thresh=25, pixel_num_thresh=2000,
                                                                 area_expand_thresh=5,
                                                                 area_same_color_thres=5,
                                                                 brighter_thresh=5
                                                                 )
        abnormal_masks[i] = torch.tensor(abnormal_mask,device = point_pixels.device).bool()



        # 4. update point_validation
        temp_point_to_remove_mask = abnormal_masks[i][point_pixels[i][point_validation[i]][:,0],point_pixels[i][point_validation[i]][:,1]]
        new_point_validation[i][point_validation[i]] = torch.logical_and(
            point_validation[i][point_validation[i]],
            torch.logical_not(temp_point_to_remove_mask))
        end = time.time()
        print(i,end-start)
        # 5. save depth map # not put into detect_abnormal_bright_spots_in_gray_img
        # if save_path is not None:
        #     depth_map_file = os.path.join(save_path, f'{i}_depth.png')
        #     # remove abnormal in depth
        #     non_abnormal_value_mask = torch.logical_and(non_empty_pixel_mask,torch.logical_not(abnormal_masks[i]))
        #     dense = naive_inpainting(img=(sparse_depth_maps[i]).unsqueeze(0),  # [1,res,res]
        #                                        no_need_inpaint_mask2=non_abnormal_value_mask.unsqueeze(0),  # [1,res,res]
        #                                        method='nearest')[0]
        #     foreground_mask = foreground_mask.cpu().numpy()
        #     dense = (dense - min_for_norm) / (max_for_norm - min_for_norm)
        #     dense[dense < 0] = 0
        #     dense[~foreground_mask] = 0 # apply foreground_mask in depth
        #     dense=np.flip(dense,0) # flip depth map upside down
        #     depth_map_img  = np.repeat(dense[np.newaxis, ...], 3, axis=0).astype(np.float64)  # [3,res,res]
        #     save_CHW_RGB_img(depth_map_img, depth_map_file)
    return new_point_validation

# ----------------------- 2D to 3D --------------------
def project_rendered_img_2_textured_mesh1(vertices, faces, images, face_view_ids, face_vertex_uvs, save_path,
                                          hard_mask_0):
    '''
    :param vertices: [V,3]
    :param faces: [F,3]
    :param images: [img_num, 3,res,res]
    :param face_view_ids: [F]
    :param face_vertex_uvs:  [F,3,2]
    :return:
    '''
    # prepare folders
    img_num,_,res,res = images.shape
    os.makedirs(save_path,exist_ok=True)

    # write images
    for i in range(img_num):
        img_file_name = os.path.join(save_path, f'{i}.png')
        ########################## post-processing
        dilate = True
        if dilate:
            img = images[i]* 255.0
            img = img.transpose(1, 2, 0)   # to HWC
            img = img.clip(0, 255)#.cpu().numpy()

            kernel = np.ones((3,3), 'uint8')

            # mask = cv2.dilate((1 - hard_mask_0[i].transpose(1, 2, 0)), kernel, iterations=1)
            mask = (1 - hard_mask_0[i].transpose(1, 2, 0))
       
            iterations = 20
            for j in range(iterations):
                dilate_img = cv2.dilate(img, kernel,iterations=1)  #
                img = img * (1 - mask) + dilate_img * mask
                # img = dilate_img
            img = img.clip(0, 255).astype(np.uint8)
            PIL.Image.fromarray(np.ascontiguousarray(img[:, :, :]), 'RGB').save(img_file_name)
        else: # inpaint
            img = images[i]
            black_pixel_mask = img.mean(0)<10.0/255.0
          
            need_to_fill_mask = black_pixel_mask #* (1-can_be_black_pixels_mask[i]) # black but cannot be black, need to fill
            need_to_fill_mask = need_to_fill_mask.astype(bool)
         

            # inpaint
            use_cv2 = False
            if use_cv2:
                # Convert the mask to uint8
                mask_uint8 = need_to_fill_mask.astype('uint8')

                # Initialize an empty array to hold the filled image
                filled_img = np.empty_like(img)

                # Loop over each color channel
                for i in range(img.shape[0]):
                    # Apply inpainting to this color channel
                    filled_img[i] = cv2.inpaint(img[i], mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                    # filled_img[i] = cv2.inpaint(img[i], mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_NS)

                # filled_img is now the image with 'need_to_fill' pixels filled in
                save_CHW_RGB_img(filled_img, img_file_name)
            else:
                # Create a grid of pixel coordinates
                y_coords, x_coords = np.indices(img.shape[1:])
                coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

                # Flatten the image and mask arrays
                img_flat = img.reshape(img.shape[0], -1)
                mask_flat = need_to_fill_mask.ravel()

                # Filter the image array to only include valid pixels
                valid_pixels = img_flat[:, ~mask_flat]
                valid_coords = coords[~mask_flat]
                x = np.arange(res)
                y = np.arange(res)
                xx, yy = np.meshgrid(x, y, indexing='xy') # xy or ij


                interpolated_pixels = griddata(valid_coords, valid_pixels.T, (xx,yy), method='nearest') # res,res,3
        
                filled_img = interpolated_pixels.transpose(2,0,1)
             
                save_CHW_RGB_img(filled_img, img_file_name)



        ##########################
        # save_CHW_RGB_img(img,img_file_name)
        # print('save',img_file_name)

    # write .mtl file
    mtl_file_name = os.path.join(save_path, f'model_normalized.mtl')  # matname = '%s/%s.mtl' % (fol, na)
    with open(mtl_file_name, 'w') as fid:
        for i in range(img_num):
            fid.write(f'newmtl material_{i}\n')
            fid.write('Kd 1 1 1\n')
            fid.write('Ka 0 0 0\n')
            fid.write('Ks 0.4 0.4 0.4\n')
            fid.write('Ns 10\n')
            fid.write('illum 2\n')
            fid.write('map_Kd %d.png\n' % (i))
            fid.write('\n')


    # write .obj file
    obj_file_name = os.path.join(save_path, f'model_normalized.obj')  # matname = '%s/%s.mtl' % (fol, na)
    all_face_ids = np.arange(len(faces))

    with open(obj_file_name, 'w') as fid:
        fid.write('mtllib model_normalized.mtl\n' )
        # write v
        for v in vertices:
            fid.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        # write vt
        for vt in face_vertex_uvs.reshape(-1,2):
            fid.write('vt %f %f\n' % (vt[0], vt[1]))
        # write f
        for i in range(img_num):
        # for i in range(img_num-1,-1,-1):

            fid.write(f'usemtl material_{i}\n')
            for face_id in all_face_ids[face_view_ids == i]:
                f_v_id = faces[face_id] +1
                f_vt_id = np.array([face_id *3,face_id *3+1,face_id *3+2] ) +1
                fid.write('f %d/%d %d/%d %d/%d\n' % (f_v_id[0], f_vt_id[0], f_v_id[1], f_vt_id[1], f_v_id[2], f_vt_id[2]))

    # print('save', mtl_file_name)

# ----------------------- dealing with 2D --------------------
def paint_pixels(img, pixel_coords, pixel_colors, point_size):
    '''
    :param img: torch tensor of shape [3,res,res]
    :param pixel_coords: [N,2]
    :param pixel_colors: [N,3]
    :param point_size: paint not only the given pixels, but for each pixel, paint its neighbors whose distance to it is smaller than (point_size-1).
    :return:
    '''


    N = pixel_coords.shape[0]
    C = img.shape[0]

    if not torch.is_tensor(pixel_colors):
        pixel_colors = pixel_colors * torch.ones((N,C),device=img.device).float()
    if point_size == 1:
        img[:, pixel_coords[:, 0], pixel_coords[:, 1]] = pixel_colors.permute(1, 0)
    else:

        pixel_coords = pixel_coords.long()
        if point_size > 1:
            xx, yy = torch.meshgrid(torch.arange(-point_size + 1, point_size, 1),
                                    torch.arange(-point_size + 1, point_size, 1))

            grid = torch.stack((xx, yy), 2).view(point_size * 2 - 1, point_size * 2 - 1, 2).to(img.device) # grid_res,grid_res,2

            grid_res = grid.shape[0]
            grid = grid + pixel_coords.unsqueeze(1).unsqueeze(1) # [N,grid_res,grid_res,2]


            pixel_colors = pixel_colors.unsqueeze(1).unsqueeze(1).repeat(1,grid_res,grid_res,1) # [N,3] -> [N,grid_res,grid_res,3]
            mask = (grid[:, :,:, 0] >= 0) & (grid[:, :,:,  0] < img.shape[1]) & (grid[:, :, :, 1] >= 0) & (
                    grid[:, :,:,  1] < img.shape[2])  # [N,grid_res,grid_res],

            grid = grid[mask]    # [final_pixel_num,2】
            pixel_colors = pixel_colors[mask] # [final_pixel_num,3】
            indices = grid.long()
            img[:, indices[:, 0], indices[:, 1]] = pixel_colors.permute(1,0)#.unsqueeze(1).repeat(1, grid.shape[0], 1)

    return img

def get_forground_inner_edge_mask(foreground_mask,method = 'dilate'):
    '''
    :param foreground_mask: res,res
    :param method: 'shift' or 'dilate'. 'dilate' produces more smooth_depth_map result
    :return:
    '''
    if method == 'shift':
        # Create masks for the neighbors in each direction
        top_mask = torch.roll(foreground_mask, shifts=1, dims=0)
        bottom_mask = torch.roll(foreground_mask, shifts=-1, dims=0)
        left_mask = torch.roll(foreground_mask, shifts=1, dims=1)
        right_mask = torch.roll(foreground_mask, shifts=-1, dims=1)

        # Find the edge pixels by checking if any neighbor is 0
        # edge_mask = (top_mask | bottom_mask | left_mask | right_mask) & foreground_mask
        top_edge_mask = foreground_mask ^ (top_mask& foreground_mask)
        bottom_edge_mask = foreground_mask ^ (bottom_mask& foreground_mask)
        left_edge_mask = foreground_mask ^ (left_mask& foreground_mask)
        right_edge_mask = foreground_mask ^ (right_mask& foreground_mask)
        edge_mask = top_edge_mask | bottom_edge_mask | left_edge_mask | right_edge_mask
        edge_mask = edge_mask * foreground_mask

    elif method == 'dilate':
        dilated_back_mask = torch.nn.functional.max_pool2d((~foreground_mask).unsqueeze(0).unsqueeze(0).float(),
                                                      kernel_size=3, stride=1, padding=1).squeeze().bool() # res,res
        edge_mask = dilated_back_mask & foreground_mask


    vis = False
    if vis:
        foreground_mask_img = foreground_mask.unsqueeze(0).repeat(3, 1, 1)
        edge_mask_img = edge_mask.unsqueeze(0).repeat(3, 1, 1)
        cat = cat_images(foreground_mask_img.cpu().numpy(), edge_mask_img.cpu().numpy())

        display_CHW_RGB_img_np_matplotlib(cat)
    return edge_mask


def paint_pixels_by_mask_ratio(img,pixel_coords,pixel_colors,mask_ratio,hard_mask):
    '''
    :param img:
    :param pixel_coords:
    :param pixel_colors:
    :param mask_ratio:
    :param hard_mask: [res,res,1]
    :return:
    '''
    N = pixel_coords.shape[0]
    res = hard_mask.shape[0]
    device = pixel_coords.device
    target_painted_pixel_num = ((hard_mask.sum() * mask_ratio)).long()
    if not torch.is_tensor(pixel_colors):
        pixel_colors = pixel_colors * torch.ones((N, 3), device=img.device).float()

    # calculate point size and extra_pixels to paint
    occupied_pixel_num = pixel_coords.shape[0]#.float()
    forground_pixel_num = (hard_mask).sum().float()
    point_size_1_mask_ratio = 1-occupied_pixel_num/forground_pixel_num


    # get candidate points
    mask_img = hard_mask.repeat(1, 1, 3).permute(2, 0, 1).float()
    hard_mask2_point_size1 = 1 - torch.clone(mask_img)
    hard_mask2_point_size2 = 1 - torch.clone(mask_img)
    hard_mask2_point_size1 = paint_pixels(hard_mask2_point_size1, pixel_coords, 1.0, point_size=1)
    hard_mask2_point_size2 = paint_pixels(hard_mask2_point_size2, pixel_coords, 1.0, point_size=2)


    # calculate ratio
    occupied_pixel_num2 = (hard_mask2_point_size2[0,:,:] * hard_mask[:,:,0]).sum().float()
    point_size_2_mask_ratio = 1 - occupied_pixel_num2 / forground_pixel_num
    # print('-' * 50)
    # print('point_size_1_mask_ratio', point_size_1_mask_ratio)
    # print('point_size_2_mask_ratio', point_size_2_mask_ratio)
    # print('-' * 50)
    if point_size_2_mask_ratio < mask_ratio:
        paint_back2empty_pixel_num = occupied_pixel_num2 - target_painted_pixel_num
        # print('-' * 50)
        # print('occupied_pixel_num2', occupied_pixel_num2)
        # print('target_painted_pixel_num', target_painted_pixel_num)
        # print('paint_back2empty_pixel_num', paint_back2empty_pixel_num)
        # print('-' * 50)
        paint_back2empty_pixel_num = paint_back2empty_pixel_num.long()
        candidate_mask = hard_mask2_point_size2 - hard_mask2_point_size1
        candidate_mask = candidate_mask[0, :, :].bool()


        temp = torch.where(candidate_mask == True)
        candidate_pixel_coords = torch.vstack((temp[0], temp[1])).permute(1, 0)

        candidate_pixel_ids = np.arange(candidate_pixel_coords.shape[0])
        np.random.shuffle(candidate_pixel_ids)
        selected_candidate_pixel_ids = candidate_pixel_ids[:paint_back2empty_pixel_num]
        hard_mask2_point_size2 = paint_pixels(hard_mask2_point_size2,
                                              candidate_pixel_coords[selected_candidate_pixel_ids],0,point_size=1)
        sparse_img = torch.zeros((3, res, res), device=device)
        sparse_img = paint_pixels(sparse_img, pixel_coords,pixel_colors, point_size=2)
        sparse_img = paint_pixels(sparse_img, candidate_pixel_coords[selected_candidate_pixel_ids],0, point_size=1)

    else:
        sparse_img = torch.zeros((3, res, res), device=device)
        sparse_img = paint_pixels(sparse_img, pixel_coords, pixel_colors, point_size=2)
    check = False
    if check:
        occupied_pixel_num2 = (hard_mask2_point_size2[0, :, :] * hard_mask[:, :, 0]).sum().float()
        point_size_2_mask_ratio = 1 - occupied_pixel_num2 / forground_pixel_num

        print('-'*50)
        print('point_size_2_mask_ratio',point_size_2_mask_ratio)
        print('-' * 50)
    return sparse_img, hard_mask2_point_size2


def naive_inpainting(img,no_need_inpaint_mask2,method='linear'):
    '''
    :param img: C,H,W
    :param no_need_inpaint_mask2:
    :param method: 'linear' or 'nearest'
    :return:
    '''
    img = img.cpu().numpy()
    no_need_inpaint_mask2 = no_need_inpaint_mask2.cpu().numpy()
    res = img.shape[1]
    no_need_inpaint_mask2 = no_need_inpaint_mask2[0]
    need_to_fill_mask = ~(no_need_inpaint_mask2.astype(np.bool_))
    # Create a grid of pixel coordinates
    y_coords, x_coords = np.indices(img.shape[1:])
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Flatten the image and mask arrays
    img_flat = img.reshape(img.shape[0], -1)
    mask_flat = need_to_fill_mask.ravel().astype(np.bool_)

    # Filter the image array to only include valid pixels
    valid_pixels = img_flat[:, ~mask_flat]
    valid_coords = coords[~mask_flat]


    x = np.arange(res)
    y = np.arange(res)
    xx, yy = np.meshgrid(x, y, indexing='xy')  # xy or ij

    interpolated_pixels = griddata(valid_coords, valid_pixels.T, (xx, yy), method=method)  # res,res,3 # linear, nearest

    filled_img = interpolated_pixels.transpose(2, 0, 1)
    # save_CHW_RGB_img(filled_img, img_file_name)
    return filled_img



def naive_inpainting_torch(img, no_need_inpaint_mask, method='linear'):
    """ Not implemented
    :param img: B, C, H, W (C=3)
    :param no_need_inpaint_mask: B, C, H, W (C=3)
    :param method: 'linear' or 'nearest'
    :return: filled_img: B, C, H, W
    """
    raise NotImplementedError


#-----------------------------------  assign utils -----------------------------------

def assign_labels_to_invalid_by_most_neighbors(neighbors,all_labels,label_num):
    '''
    Assign labels to invalid faces (whose labels are currently -1) by most of their valid neighbors.
    :param neighbors: [face_num,3]. Each face can have at most 3 adjacent neighbors. If not 3, values are -1.
    :param all_labels: [face_num]. Invalid faces has labels to be 1.
    :param label_num:
    :return:
    '''
    invalid_mask = all_labels == -1
    invalid_point_num = invalid_mask.sum()
    invalid_neighbors = neighbors[invalid_mask] # [invalid_num, 3], 3 means each face has at most 3 neighbors
    neighbor_valid_mask = invalid_neighbors >-1

    invalid_face_neighbor_labels = all_labels[invalid_neighbors] # [invalid_num, 3]


    # invalid_assigned_labels = invalid_neighbor_labels.mode(1)[0] #if directly use this, there will still be empty faces


    # to ignore invalid neighbors:
    invalid_face_neighbor_labels[~neighbor_valid_mask] = label_num  # set labels for invalid neighbors to label_num

    # temp is a table of shape [invalid_point_num,label_num+1], calculate frequency of each label. invalid is put to the last dimension
    temp = torch.zeros((invalid_point_num, label_num + 1), device=all_labels.device)  # [invalid_point_num,label_num+1]
    K = 3
    for i in range(K):
        # temp: N,20;   neighbor_labels[:,i]: [N]
        temp[torch.arange(invalid_point_num, device=all_labels.device), invalid_face_neighbor_labels[:, i]] += 1
        # print(temp)
    temp = temp[:, :label_num]
    still_invalid_point_mask = temp.sum(1) == 0
    most_common_labels = temp.max(1)[1]
    most_common_labels[still_invalid_point_mask] = -1
    # print('most_common_labels.shape',most_common_labels.shape)
    # print('most_common_labels',most_common_labels.max())
    # print('still_invalid_point_mask.sum()',still_invalid_point_mask.sum())

    # return most_common_labels


    all_labels[invalid_mask] = most_common_labels
    return all_labels


def smooth_labels_by_neighbors(neighbors,all_labels):
    neighbor_labels = all_labels[neighbors] # [face_num,3]
    all_neighbor_same_label_mask = neighbor_labels.max(1)[0] == neighbor_labels.min(1)[0] # [face_num]
    all_labels[all_neighbor_same_label_mask] = neighbor_labels[all_neighbor_same_label_mask][:,0]
    return all_labels



#-----------------------------------  neighbor adjacency utils -----------------------------------

def create_neighbors_tensor(N, adjacency_pairs):
    '''
    :param N:
    :param adjacency_pairs: [M,2]. adjacency_pairs[i,0] and adjacency_pairs[i,1] are neighbors
    :return: [N,K], indicating the K neighbors of N input points. Note that when a point has fewer than K neighbors, fill it with -1
    '''
    device = adjacency_pairs.device
    adjacency_pairs_repeat = torch.cat([adjacency_pairs[:,[1,0]],adjacency_pairs],0) # [2*M,2]
    neighbor_counts = torch.bincount(adjacency_pairs.view(-1)) # [max_index_of_not_alone_elements+1] mostly like to = N, but can be smaller. indicating
    # print('neighbor_counts',neighbor_counts)
    if neighbor_counts.shape[0] < N: # pad zeros
        neighbor_counts = torch.cat([neighbor_counts, torch.zeros((N-neighbor_counts.shape[0]),device=device)],dim=0).long() # [N]

    K = torch.max(neighbor_counts).item()


    index_1_to_N = torch.arange(N,device=device).long() # .to(device)



    add = []
    print('K',K)
    for i in range(K):
        index_i_neighbor = index_1_to_N[neighbor_counts == i]  # [...] # the ids of points that has i neighbors
        if index_i_neighbor.shape[0] > 0:
            add_i = torch.cat(
                [index_i_neighbor.unsqueeze(-1), -torch.ones((index_i_neighbor.shape[0], 1), device=device)],
                dim=1).long()  # [...,1], [...,1] -> [..., 2 ]
            add_i = add_i.repeat(K-i, 1)  # [...*(i+1),2]
            add.append(add_i)
            

    if len(add) >0:
        add = torch.cat(add, dim=0)  # [N*sum(i+1 for i in range(K)),2]

        adjacency_pairs_repeat = torch.cat([adjacency_pairs_repeat, add], dim=0)  # [N*sum(i+1 for i in range(K)),2]

    # sort adjacency_pairs_repeat by the [:,0]
    adjacency_pairs_repeat = adjacency_pairs_repeat[adjacency_pairs_repeat[:, 0].sort()[1]] # [N*3,2]


    neighbors = adjacency_pairs_repeat[:,1].reshape(N,K)

    return neighbors



def get_face_view_pixel_num(face_idxs,cam_num,face_num):
    '''
    :param face_idxs: [cam_num,res,res] per pixel face id
    :return face_view_pixel_num:
    '''
    device = face_idxs.device
    face_view_pixel_num = torch.zeros((face_num, cam_num), device=device).long()

    face_view_pixel_num = -1 * torch.ones((cam_num, face_num),device=device)
    for i in range(cam_num):

        cur_face_id = 0
        batch_face_num = 2000
        
        while cur_face_id < face_num - 1:
            # Use min function to handle the last batch
            end_face_id = min(cur_face_id + batch_face_num, face_num)
            mask = (face_idxs[i].unsqueeze(0) == torch.arange(cur_face_id, end_face_id)
                    .unsqueeze(-1).unsqueeze(-1).to(device))

            face_view_pixel_num[i][cur_face_id:end_face_id] = mask.sum(dim=-1).sum(dim=-1)
            cur_face_id += batch_face_num
        
    return face_view_pixel_num


def assign_face_view(faces,neighbors,face_idxs,similarity,view_num):
    device = faces.device
    face_view_ids = -1 * torch.ones((len(faces)), device=device).long()

    # calculate how many pixels are there for each face in each view
    face_view_pixel_num = get_face_view_pixel_num(face_idxs,cam_num=view_num,face_num=len(faces)) # cam_num, face_num
    # face_view_pixel_num = get_face_view_pixel_num(face_idxs,cam_num=len(cams),face_num=len(faces)) # cam_num, face_num

    # 0. assign visible faces to views by normal; visible view prior to invisible view
    face_view_valid = face_view_pixel_num > 0  # [cam_num,face_num]
    face_view_valid = face_view_valid.permute(1, 0)  # [face_num,cam_num]
    face_visible_mask = face_view_valid.max(1)[0].bool()  # [face_num]

    similarity[~(face_view_valid.bool())] -= 100000
    face_temp_view_ids = torch.argmax(torch.softmax(similarity, dim=1), dim=1)  # [face_num,]
    face_view_ids[face_visible_mask] = face_temp_view_ids[face_visible_mask]
    # face_view_ids[~face_visible_mask] = 0
    valid_face_num = face_view_valid.max(1)[0].bool().sum()
    directly_assign_face_ratio = valid_face_num / len(faces)


    # 1. if a face has a view with most pixels of the face, then assign the face to this view (instead of by normal)
    # per_face_max_pixel_num, per_face_max_pixel_view_id = face_view_pixel_num.max(0)
    # unique_rows = torch.unique_consecutive(face_view_pixel_num, dim=0)
    # unique_max_values = unique_rows[-1]
    # is_max_unique =  torch.eq(per_face_max_pixel_num, unique_max_values)
    # face_view_ids[is_max_unique] = per_face_max_pixel_view_id[is_max_unique]
    # directly_assign_face_ratio = is_max_unique.sum() / len(faces)
    # # print('directly_assign_face_ratio',directly_assign_face_ratio)
    # # print('face_view_ids',face_view_ids)


    # 2. Assign invisible faces by their neighbors
    face_view_ids = assign_labels_to_invalid_by_most_neighbors(neighbors, face_view_ids,
                                                               label_num=view_num)

    invalid_num = (face_view_ids < 0).sum()
    last_invalid_num = invalid_num + 1
 
    while invalid_num > 0 and invalid_num != last_invalid_num:
        last_invalid_num = invalid_num.item()
        face_view_ids = assign_labels_to_invalid_by_most_neighbors(neighbors, face_view_ids,
                                                                   label_num=view_num)
        invalid_num = (face_view_ids < 0).sum()
        

    # smooth_depth_map assignment
    face_view_ids = smooth_labels_by_neighbors(neighbors, face_view_ids)
    face_view_ids = smooth_labels_by_neighbors(neighbors, face_view_ids)
    face_view_ids = smooth_labels_by_neighbors(neighbors, face_view_ids)
    # print('face_view_ids',face_view_ids.min())
    return face_view_ids

# ------------------------------------------ other stuff
def get_face_vertice_uvs(faces,face_view_ids,vertex_uvs,view_num):
    device = faces.device
    face_vertex_uvs = torch.zeros((len(faces), 3, 2), device=device)
    # get uv
    for i in range(view_num):
        face_vertex_uvs[face_view_ids == i] = vertex_uvs[i][faces[face_view_ids == i]]
    return face_vertex_uvs

def get_sparse_images(point_pixels,colors,point_validation,hard_masks,save_path,
                      view_num,res,point_size,edge_point_size,mask_ratio_thresh):
    device = point_pixels.device
    sparse_imgs = torch.zeros((view_num,3,res,res),device=device)
    hard_mask2s = torch.zeros((view_num,3,res,res),device=device)
    hard_mask0s = torch.zeros((view_num,3,res,res),device=device)
    scale_factors = torch.zeros((view_num),device=device)
    for i in range(view_num):
        if save_path is not None:
            sparse_img_file = os.path.join(save_path, f'{i}_sparse.png')
            mask0_img_file = os.path.join(save_path, f'{i}_mask0.png')
            mask2_img_file = os.path.join(save_path, f'{i}_mask2.png')

        sparse_img,hard_mask0,hard_mask2,mask_ratio,scale_factor = get_one_sparse_img(point_pixels[i],colors,point_validation[i],hard_masks[i],
                                   res,point_size,edge_point_size,mask_ratio_thresh=mask_ratio_thresh)



        sparse_img = sparse_img * hard_mask0  # [3,res,res] * [3,res,res]
        # hard_mask2 = hard_mask2 * hard_mask0  # [3,res,res] * [3,res,res] # don't do this

        sparse_imgs[i] = sparse_img
        hard_mask0s[i] = hard_mask0
        hard_mask2s[i] = hard_mask2
        scale_factors[i] = scale_factor
        if save_path is not None:
            save_CHW_RGB_img(sparse_img.cpu().numpy(),sparse_img_file)
            save_CHW_RGB_img(hard_mask0.cpu().numpy(),mask0_img_file)
            save_CHW_RGB_img(hard_mask2.cpu().numpy(),mask2_img_file)

    return sparse_imgs,hard_mask0s,hard_mask2s,scale_factors

def get_inpainted_images(sparse_imgs,hard_masks,hard_mask2s,save_path,inpainter,view_num,method = 'linear'):
    device = sparse_imgs.device
    inpainted_imgs = torch.zeros_like(sparse_imgs).to(device)
    if method =='diff_inpaint':
        batch_inpaint = True
        if batch_inpaint:
            print('start inpainting')
            start = time.time()
            inpainted_imgs = inpainter.inpaint(masked_imgs=sparse_imgs.permute(0,2, 3, 1).cpu().numpy(),
                                              masks=hard_mask2s.permute(0,2, 3, 1).cpu().numpy())
            end = time.time()
            print('inpainting time', end - start, 's for',view_num,'images')  #
            # inpainted_imgs = torch.stack(inpainted_imgs,)
            for i in range(view_num):
                inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
                save_CHW_RGB_img(inpainted_imgs[i].cpu().numpy(), inpainted_img_path)
        else:
            total_time = 0
            for i in range(view_num):
                start = time.time()
                inpainted_img = inpainter.inpaint(masked_imgs=sparse_imgs[i].permute(1, 2, 0).unsqueeze(0).cpu().numpy(),
                                                  masks=hard_mask2s[i].permute(1, 2, 0).unsqueeze(0).cpu().numpy())[0]
                end = time.time()
                total_time += end - start
                print('inpainting time', end - start, 's')  #
                inpainted_imgs[i] = inpainted_img
                if save_path is not None:
                    inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
                    save_CHW_RGB_img(inpainted_img.cpu().numpy(), inpainted_img_path)
            print('inpainting time', total_time, 's for',view_num,'images')
    elif method == 'DDNM_inpaint':
        total_time = 0
        for i in range(view_num):
            start = time.time()
            inpainted_img = inpainter.inpaint(masked_imgs=sparse_imgs[i].permute(1, 2, 0).unsqueeze(0),
                                              masks=hard_mask2s[i].permute(1, 2, 0).unsqueeze(0))[0]
            end = time.time()
            total_time += end - start
            print(i,'/',view_num,'inpainting time', end - start, 's')  #
            inpainted_imgs[i] = inpainted_img
            if save_path is not None:
                inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
                save_CHW_RGB_img(inpainted_img.cpu().numpy(), inpainted_img_path)
        print('inpainting time', total_time, 's for', view_num, 'images')
    elif method == 'linear' or 'nearest':
        for i in range(view_num):
            start = time.time()
            inpainted_img = naive_inpainting(img=sparse_imgs[i], no_need_inpaint_mask2=hard_mask2s[i],method=method)
            inpainted_img = torch.tensor(inpainted_img, device=device)

            end = time.time()
            print('inpainting time', end - start, 's')  #
            inpainted_imgs[i] = inpainted_img
            if save_path is not None:
                inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
                save_CHW_RGB_img(inpainted_img.cpu().numpy(), inpainted_img_path)

        # start = time.time()
        # inpainted_imgs = naive_inpainting_torch(sparse_imgs,hard_mask2s,method=method)
        # end = time.time()
        # print('inpainting time', end - start, 's')  #
        # for i in range(view_num):
        #     if save_path is not None:
        #         inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
        #         save_CHW_RGB_img(inpainted_imgs[i].cpu().numpy(), inpainted_img_path)
    return inpainted_imgs


def get_one_sparse_img(point_pixels,colors,point_validation,hard_mask,res,point_size,edge_point_size,
                       mask_ratio_thresh =  0.82 # 0.82 useful;   100.0: no thresh # here is a thresh pay attention!!
                       ):
    '''

    :param point_pixels:
    :param colors:
    :param point_validation:
    :param hard_mask:
    :return:
    '''
   
    # get mask ratio (if too big, need to rescale the image to smaller to decrease mask ratio for better inpainting
    forground_pixel_num = (hard_mask).sum().float()
    valid_pixel_num = point_validation.sum()
    mask_ratio = 1 - valid_pixel_num / forground_pixel_num
    # print('mask_ratio',mask_ratio)

    if mask_ratio > mask_ratio_thresh:
        wanted_foreground_pixel_num = valid_pixel_num / (1-mask_ratio_thresh)
        scale_factor = wanted_foreground_pixel_num / forground_pixel_num # smaller than 1
        point_uvs = point_pixels/res # between 0 and 1
        point_uvs = point_uvs *2-1 # between -1 and 1
        point_uvs = point_uvs * scale_factor
        point_uvs = (point_uvs+1) * 0.5 # between 0 and 1
        point_pixels = point_uvs * res # between 0 and res
        point_pixels = point_pixels.clip(0,res-1)
        point_pixels = point_pixels.long()
        after_res = (res* scale_factor).floor().int()

        if (res - after_res) %2 == 1:
            after_res = after_res +1
        pad_value = (res-after_res)/2
        pad_value = pad_value.int().item()

        transform = transforms.Compose([
            transforms.Resize((after_res, after_res)),
            transforms.Pad((pad_value,pad_value), fill=0),

        ])

        hard_mask = transform(hard_mask.unsqueeze(0)).squeeze(0).bool() # hard_mask is [res,res]


    else:
        scale_factor = 1


    sparse_img = torch.zeros((3, res, res), device=point_pixels.device)
    valid_point_pixels = point_pixels[point_validation]
    sparse_img = paint_pixels(sparse_img, valid_point_pixels, colors[point_validation],
                             point_size=point_size)

    # also paint edge pixels in addition to valid pixels
    forground_inner_edge_mask = get_forground_inner_edge_mask(hard_mask)
    foreground_inner_edge_pixel_coords = torch.nonzero( forground_inner_edge_mask)  # [edge_pixel_num,2]
    ones_tensor1 = torch.ones_like(foreground_inner_edge_pixel_coords[..., :1])
    ones_tensor2 = torch.ones_like(valid_point_pixels[..., :1])

    distance, idx = kal.metrics.pointcloud.sided_distance(
        p1=torch.cat([foreground_inner_edge_pixel_coords, ones_tensor1], dim=1).unsqueeze(0),
        p2=torch.cat([valid_point_pixels, ones_tensor2], dim=1).unsqueeze(0))
    # 1,p1_points_num
    idx = idx[0]
    foreground_inner_edge_pixel_colors = colors[point_validation][idx]
    paint_edge=True # False only for debugging
    if paint_edge:
        sparse_img = paint_pixels(sparse_img, foreground_inner_edge_pixel_coords,
                                  foreground_inner_edge_pixel_colors,
                                  point_size=edge_point_size)
    # paint hard_mask2: mask of valid points
    hard_mask0 = hard_mask.unsqueeze(-1).repeat(1, 1, 3).permute(2, 0, 1).float()
    hard_mask2 = 1 - torch.clone(hard_mask0)
    # hard_mask1 = paint_pixels(hard_mask1, valid_point_pixels, 1.0, point_size=point_size)
    hard_mask2 = paint_pixels(hard_mask2, valid_point_pixels, 1.0, point_size=point_size)
    if paint_edge:
        hard_mask2 = paint_pixels(hard_mask2, foreground_inner_edge_pixel_coords, 1.0,
                                  point_size=edge_point_size)
    ##################  calculate mask ratio ##############

    occupied_pixel_num= (hard_mask2[0,:,:] * hard_mask[:,:]).sum().float()
    forground_pixel_num = (hard_mask).sum().float()
    mask_ratio = 1-occupied_pixel_num/forground_pixel_num



    ######################################################
    sparse_img = torch.flip(sparse_img,dims = [1])
    hard_mask0 = torch.flip(hard_mask0,dims = [1])
    hard_mask2 = torch.flip(hard_mask2,dims = [1])
    return sparse_img,hard_mask0,hard_mask2,mask_ratio,scale_factor

def recon_one_shape(name,root_path,cls_id,
                    load_exist_dense_img_path,
                    coords,colors,vertices,faces,f_normals,
                    project2mesh,sample_num,
                    unsmoothed_vertices,unsmoothed_faces,
                    use_GT_multi_view_img,
                    device,
                    view_num,res,cam_res,base_dirs,cams,cam_RTs,cam_K,refine_res,eye_positions, up_dirs,

                    point_validation_by_o3d,refine_point_validation_by_remove_abnormal_depth,hidden_point_removal_radius,


                    texture_gen_method,point_size,edge_point_size,

                    crop_img, crop_padding,mask_ratio_thresh,
                    unproject_by, optimize_from,# face, vertex
                    edge_dilate_kernels,
                    naive_face_view,

                    inpainter,glctx,
                    logger,

                    xatlas_texture_res = 1024,


                    **kwargs):
    with torch.no_grad():
        # print('cams[0].intrinsics.projection_matrix()',cams[0].intrinsics.projection_matrix().shape,cams[0].intrinsics.projection_matrix())
        ''' Skip exist'''
        save_path = os.path.join(root_path,'meshes', cls_id, name, 'models')  # f'temp/samples/textured_mesh/{name}/{name}'
        obj_file = os.path.join(save_path, 'model_normalized.obj')
        mlt_file = os.path.join(save_path, 'model_normalized.mtl')
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(obj_file) and os.path.exists(mlt_file):
            print(f'skip exist {obj_file}')
            return
        else:
            print('no exist:', obj_file)
            print('no exist:', mlt_file)



        ''' Vis input '''
        # vis = False
        # if vis:
            # from utils.vtk_basic import vis_actors_vtk, get_colorful_pc_actor_vtk, get_mesh_actor_vtk
            # mesh_tri = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())
            # vis_actors_vtk([
            #     get_mesh_actor_vtk(mesh_tri),
            #     get_colorful_pc_actor_vtk(coords.cpu().numpy(), colors)
            # ]
            # )
            # return
            # mesh = o3d.geometry.TriangleMesh()
            # mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            # mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            # o3d.io.write_triangle_mesh(f'weird_{name}.obj', mesh)




        '''project pc to mesh (in the final version we don't do this)'''
        if project2mesh:
            projected_points, points_face_ids = \
                project_poitns2mesh_by_sample_CD(points=coords, mesh_verts=vertices, mesh_faces=faces,
                                                 sample_num=sample_num)
        else:
            projected_points = coords

        vis=False
        if vis:
            from utils.vtk_basic import vis_actors_vtk,get_pc_actor_vtk,get_colorful_pc_actor_vtk
            vis_actors_vtk([
                get_pc_actor_vtk(projected_points.cpu().numpy(),color = (0,0,1),point_size=3,opacity=0.3),
                get_pc_actor_vtk(coords.cpu().numpy(),color = (0,1,0),point_size=3,opacity=0.3),
                # get_colorful_pc_actor_vtk(projected_points.cpu().numpy(),
                #                           point_colors=colors.cpu().numpy(),point_size=3,opacity=0.3),
                # get_pc_actor_vtk(vertices.cpu().numpy(),color = (1,0,0),point_size=3,opacity=0.7),

            ])

        ''' Prepare data for projecting (rendering) 3D to 2D'''


        start_project = time.time()
        hard_masks, face_idxs, mesh_normalized_depths, vertice_uvs, uv_centers, uv_scales,padding,point_uvs,point_depths = \
            get_rendered_hard_mask_and_face_idx_batch(cams, vertices, faces, projected_points,glctx=glctx,
                                                      rescale=crop_img,padding=crop_padding)




        '''get muti-view images'''
        if use_GT_multi_view_img:
            mesh_file = os.path.join(
                '/home/me/RfDNet/datasets/ShapeNetCore.v2.clean_mesh', cls_id,
                name, 'models', 'model_normalized.obj')
            inpainted_images = render_textured_mesh(mesh_file, cams, device,glctx=glctx,save_path=None, save=False,
                                                    vertices=vertices,normalize_mesh=False)
            # save inpainted images and hard masks
            for i in range(len(cams)):
                hard_mask0s = torch.flip(hard_masks.float(), dims=(1,))  # [cam_num, res,res] # flip upside down
                hard_mask0s = hard_mask0s.unsqueeze(1).repeat(1, 3, 1, 1)  # [cam_num,3, res,res]

                inpainted_img_path = os.path.join(save_path, f'{i}_inpainted.png')
                mask0_img_file = os.path.join(save_path, f'{i}_mask0.png')

                save_CHW_RGB_img(hard_mask0s[i].cpu().numpy(), mask0_img_file)
                save_CHW_RGB_img(inpainted_images[i].cpu().numpy(), inpainted_img_path)
        else:

            if cam_res != res:
                hard_masks = transforms.Resize((res,res))(hard_masks.unsqueeze(1).float()).squeeze(1).bool() # N,cam_res,cam_res


            point_validation1,_ = get_point_validation_by_depth(cam_res,point_uvs,point_depths,mesh_normalized_depths)
            point_validation2 = get_point_validation_by_o3d(projected_points,eye_positions,hidden_point_removal_radius)
            point_validation = torch.logical_or(point_validation1,point_validation2)
           

            if refine_point_validation_by_remove_abnormal_depth:
                point_validation = refine_point_validation(cam_RTs,cam_K, refine_res,
                               hard_masks, point_validation, point_uvs, projected_points,save_path)

            ''' load existing multi-view images'''
            if load_exist_dense_img_path is not None:
                exist_inpainted_multiview_imgs = True
            load_path = os.path.join(load_exist_dense_img_path,'meshes', cls_id, name, 'models')
          
            for i in range(view_num):
          
                inpainted_img_path = os.path.join(load_path, f'{i}.png')
                if not os.path.exists(inpainted_img_path):
                    exist_inpainted_multiview_imgs = False



            # get sparse img
            point_pixels = point_uvs * res   # [num_cameras,piont_num,2]
            point_pixels = point_pixels.long()
            point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                                     dim=-1)  # switch x and y
            point_pixels = point_pixels.clip(0, res - 1)
            sparse_imgs,hard_mask0s,hard_mask2s,inpaint_scale_factors = get_sparse_images\
                (point_pixels,colors,point_validation,hard_masks,save_path,view_num,res,point_size,edge_point_size,
                 mask_ratio_thresh)


            logger.info(f'project: {time.time()-start_project} s')
            # get dense img
            start_inpainting = time.time()
            if exist_inpainted_multiview_imgs: # load from exist

                inpainted_images = torch.zeros((len(cams), 3, res, res)).to(device)
                for i in range(view_num):
                    # inpainted_img_path = os.path.join(load_path, f'{i}_inpainted.png')
                    inpainted_img_path = os.path.join(load_path, f'{i}.png')
                    inpainted_images[i] = load_CHW_RGB_img(inpainted_img_path).to(device)
        

            else: # inpaint now
                inpainted_images = get_inpainted_images(sparse_imgs,hard_masks,hard_mask2s,save_path, inpainter,view_num,
                                                        method=texture_gen_method)
                logger.info(f'inpainting: {time.time() - start_inpainting} s')

    ''' Unproject inpainted 2D rendered images back to 3D'''
    start_unproject = time.time()
    if unproject_by == 'face':
        with torch.no_grad():
            ### calculate face adjacency
            adjacency_pairs = trimesh.graph.face_adjacency(faces.cpu().numpy())  # [pair_num,2]
            adjacency_pairs = torch.tensor(adjacency_pairs, device=device).long()
            neighbors = create_neighbors_tensor(len(faces), adjacency_pairs)  # [face_num,3]

            ### get a view id for each face according to normal direction
            similarity = f_normals @ base_dirs.t()  # [face_num, view_num]
            face_view_id_by_normal = torch.argmax(torch.softmax(similarity, dim=1), dim=1)  # [face_num,]

            ### assign each triangle face to a view image
            if naive_face_view:
                face_view_ids = face_view_id_by_normal
            else:
                face_view_ids = assign_face_view(faces,  neighbors, face_idxs,similarity,view_num=view_num)

            # vertice_uvs: [cam_num, vertice_uv_num, 2]
            # inpaint_scale_factors: [cam_num]
            vertice_uvs = vertice_uvs *2 -1
            vertice_uvs = vertice_uvs * inpaint_scale_factors.unsqueeze(-1).unsqueeze(-1)
            vertice_uvs = (vertice_uvs +1) * 0.5

            face_vertex_uvs = get_face_vertice_uvs(faces,face_view_ids,vertex_uvs=vertice_uvs,view_num=view_num)
            ### save result
            project_rendered_img_2_textured_mesh1(vertices.cpu().numpy() / 0.9, faces.cpu().numpy(),
                                                  images=inpainted_images.cpu().numpy(),
                                                  face_view_ids=face_view_ids.cpu().numpy(),
                                                  face_vertex_uvs=face_vertex_uvs.cpu().numpy(),
                                                  save_path=save_path,

                                                  hard_mask_0=hard_mask0s.cpu().numpy())


    elif unproject_by == 'vertex':
        # vertices = vertices.cpu().numpy()
        # faces = faces.cpu().numpy()
        with torch.no_grad():
            xatlas_root_path = None
            if xatlas_root_path is not None:
                xatlas_save_path = (os.path.join(xatlas_root_path, name, 'xatlas.pth'))
                if os.path.exists(xatlas_save_path):  # load
                    xatlas_dict = torch.load(xatlas_save_path)
                    uvs = xatlas_dict['uvs']
                    mesh_tex_idx = xatlas_dict['mesh_tex_idx']
                    gb_pos = xatlas_dict['gb_pos']
                    mask = xatlas_dict['mask']
                else:  # calculate and save

                    uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap_w_face_id(
                        glctx, vertices, faces, resolution=xatlas_texture_res)
                    xatlas_dict = {'uvs': uvs, 'mesh_tex_idx': mesh_tex_idx, 'gb_pos': gb_pos, 'mask': mask}
                    os.makedirs(os.path.dirname(xatlas_save_path), exist_ok=True)
                    torch.save(xatlas_dict, xatlas_save_path)
            else:  # calculate without saving
                uvs, mesh_tex_idx, gb_pos, mask,per_atlas_pixel_face_id = \
                    xatlas_uvmap_w_face_id(glctx, vertices, faces, resolution=xatlas_texture_res)

            '''
            uvs:                per mesh vertex uv:                     [vert_num, 2]
            mesh_tex_idx:       per face uv coord id (to index uvs):    [face_num,3]
            gb_pos:             per pixel 3D coordinate:                [1,res,res,3] 
            mask:               per pixel validation:                   [1,res,res,1]
            per_pixel_face_id:  per pixel face id:                      [1,res,res]
            
            '''

            view_img_res = res
            res = xatlas_texture_res

            per_pixel_mask = mask[0,:,:,0] # [res,res]
            per_pixel_point_coord = gb_pos[0] # [res,res,3]
            per_atlas_pixel_face_id = per_atlas_pixel_face_id[0] #[1,res,res]

            per_pixel_pixel_coord = torch.zeros((res,res,2),device=device).long()
            xx, yy = torch.meshgrid(torch.arange(res).to(device), torch.arange(res).to(device))
            per_pixel_pixel_coord[:, :, 0] = xx
            per_pixel_pixel_coord[:, :, 1] = yy

            points = per_pixel_point_coord[per_pixel_mask] # [?,3] ??
            points_atlas_pixel_coord = per_pixel_pixel_coord[per_pixel_mask].long() # [?,2] ??


            # get per pixel depth and uv

            transformed_points = torch.zeros((len(cams), points.shape[0], 3), device=vertices.device)
            for i, cam in enumerate(cams):
                transformed_points[i] = cam.transform(points)
            per_view_per_point_depths = transformed_points[ ..., 2]
            per_view_per_point_uvs = transformed_points[..., :2]
            per_view_per_point_uvs = (per_view_per_point_uvs - uv_centers) / uv_scales  # now all between -0.5, 0.5


            per_view_per_point_uvs_no_scale = per_view_per_point_uvs.clone()
            per_view_per_point_uvs = per_view_per_point_uvs * inpaint_scale_factors.unsqueeze(-1).unsqueeze(-1)


            per_view_per_point_uvs = per_view_per_point_uvs * (1 - 2 * padding)  # now all between -0.45, 0.45
            per_view_per_point_uvs= per_view_per_point_uvs + 0.5  # now all between 0.05, 0.95

            per_view_per_point_uvs_no_scale = per_view_per_point_uvs_no_scale * (1 - 2 * padding)  # now all between -0.45, 0.45
            per_view_per_point_uvs_no_scale= per_view_per_point_uvs_no_scale + 0.5  # now all between 0.05, 0.95

     
            # Get per pixel visibility by depth
            per_view_per_point_visibility,_ = get_point_validation_by_depth(cam_res,per_view_per_point_uvs_no_scale,
                                          per_view_per_point_depths,mesh_normalized_depths,offset = 0.0001,
                                                                            vis=False)# [cam_num, point_num]

            start_shrink_visibility = time.time()
            shrink_per_view_visibility = True
            if shrink_per_view_visibility:


                per_atlas_pixel_per_view_visibility = torch.zeros((res,res,view_num),device=device).bool()

                per_atlas_pixel_per_view_visibility[per_pixel_mask] = per_view_per_point_visibility.permute(1,0)#.clone() # (res,res,view_num)

                use_torch = True
                if use_torch:

                    per_kernel_per_view_shrinked_per_pixel_visibility = get_shrinked_per_view_per_pixel_visibility_torch(
                        per_pixel_mask,per_atlas_pixel_per_view_visibility,
                        kernel_sizes= edge_dilate_kernels,
                        save_path = os.path.join(save_path,'shrink_per_view_edge')) # [kernel_num,view_num,res,res]
            logger.info(f'shrink visibility: {time.time() - start_shrink_visibility} s')
            # Get similarity between point normal and view_dir
            per_atlas_pixel_face_normal = f_normals[per_atlas_pixel_face_id] #res,res,3
           
            per_point_face_normal = per_atlas_pixel_face_normal[per_pixel_mask] # [?,3]


            similarity_between_point_normal_and_view_dir = per_point_face_normal @ torch.tensor(base_dirs,
                                                                                                device=device).t()  # [ point_num,view_num]

        

            # Get per view per point pixel
            per_view_per_point_pixel = per_view_per_point_uvs * view_img_res
            per_view_per_point_pixel = per_view_per_point_pixel.clip(0, view_img_res - 1)
            per_view_per_point_pixel = per_view_per_point_pixel.long()
            per_view_per_point_pixel = torch.cat((per_view_per_point_pixel[:, :, 1].unsqueeze(-1),
                                                  per_view_per_point_pixel[:, :, 0].unsqueeze(-1)),
                                     dim=-1)  # switch x and y if you ever need to query pixel coordiantes


        
     
            per_point_view_weight =    similarity_between_point_normal_and_view_dir
            per_point_view_weight[~(per_view_per_point_visibility.permute(1,0).bool())] -=100
            '''24.01.05'''
            point_num = per_point_face_normal.shape[0]
            # candidate_per_point_per_view_mask = torch.ones((point_num,view_num)).bool().to(device) # [point_num,view_num]

            # forget close2invisible ones
            shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[0]
            shrinked_per_view_per_point_visibility = \
                shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

            candidate_per_point_per_view_mask = \
                shrinked_per_view_per_point_visibility.permute(1, 0)  # [point_num,view_num]

            for i in range(1,len(edge_dilate_kernels)):
                # if a point is not visible in any view, try less tight mask
                per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)

                shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[i]
                shrinked_per_view_per_point_visibility = \
                    shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

                candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
                    torch.logical_or(
                        candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
                        shrinked_per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
                    )


            # if a point is not visible in any view, try less tight mask
            per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)
            candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
                torch.logical_or(
                    candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
                    per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
                )



            # now choose the ones with best normal similarity
            per_point_per_view_weight = torch.softmax(similarity_between_point_normal_and_view_dir,1) # [pointnum, view_num]
            per_point_per_view_weight[~candidate_per_point_per_view_mask] = -100
            point_view_ids = torch.argmax(per_point_per_view_weight, dim=1)


            single_view_atlas_imgs = torch.zeros((view_num,res, res, 3), device=device)
            single_view_atlas_masks = torch.zeros((view_num,res, res, 3), device=device).bool()
            atlas_img = torch.zeros((res, res, 3), device=device)
            per_pixel_view_id = -torch.ones((res,res),device=device).long()



            debug = True
            if debug:
                # paint colors by per pixel view id
                atlas_img_view_id = torch.zeros((res, res, 3), device=device)
                import seaborn as sns
                # from vtkmodules.vtkCommonDataModel import vtkDataObject

                palette = np.array(sns.color_palette("hls", len(cams))) # [color_num,3]
                palette = torch.tensor(palette,device=device).float()

                for i in range(len(cams)):
                    # paint each pixel in the atlas by its view id
                    point_this_view_mask = point_view_ids == i
                    atlas_img_view_id[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                              points_atlas_pixel_coord[point_this_view_mask][:, 1]] = palette[i]
                    # save each view with colorful background
                    view_img = inpainted_images[i].detach().clone() # [3,res,res]
                    background_mask = ~hard_mask0s[i][0].bool()  # [cam_num,3,res,res]->[res,res]
                    view_img[:,background_mask] = palette[i].unsqueeze(1) # # [3,?]= [3,1]
                    save_CHW_RGB_img(view_img.detach().cpu().numpy(),os.path.join(save_path,f'{i}.png'))

            # paint each pixel in the atlas by the query color in each view img
            for i in range(len(cams)):

                point_this_view_mask = point_view_ids == i

                view_img = inpainted_images[i]
                view_img = torch.flip(view_img,[1]) # flip upside down
                view_img = view_img.permute(1,2,0) # HWC

                atlas_img[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                          points_atlas_pixel_coord[point_this_view_mask][:, 1]] = \
                    view_img[per_view_per_point_pixel[i][point_this_view_mask][:,0],
                             per_view_per_point_pixel[i][point_this_view_mask][:,1]]

                per_pixel_view_id[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                          points_atlas_pixel_coord[point_this_view_mask][:, 1]] = i


                ################
                per_view_per_point_visibility
                shrinked_per_view_per_point_visibility
                all_visible_point_this_view_mask = shrinked_per_view_per_point_visibility.bool()[i]
                single_view_atlas_imgs[i][points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 0],
                          points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 1]] = \
                    view_img[per_view_per_point_pixel[i][all_visible_point_this_view_mask][:,0],
                             per_view_per_point_pixel[i][all_visible_point_this_view_mask][:,1]]


                single_view_atlas_masks[i][points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 0],
                                          points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 1]] = True
                single_view_atlas_imgs[i][~single_view_atlas_masks[i][..., 0]] = palette[i]

                save_CHW_RGB_img(single_view_atlas_imgs[i].permute(2,0,1).detach().cpu().numpy()[:, ::-1, :],
                                 os.path.join(root_path,'meshes',cls_id, name, 'models',f'atlax_view_{i}.png'))


            # dialate # without this, face edges will look weird
            tex_map = atlas_img
            lo, hi = (0, 1)
            img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            dilate_mask = mask[0]  # from [1,res,res,1] to [1,res,res]
            img *= dilate_mask.detach().cpu().numpy()  # added by PointDreamer author, necessary to enable later dialate
            img = img.clip(0, 255)
            dilate_mask = np.sum(img.astype(float), axis=-1,
                          keepdims=True)  # mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
            dilate_mask = (dilate_mask <= 3.0).astype(float)  # mask = (mask <= 3.0).astype(np.float)
            kernel = np.ones((3, 3), 'uint8')
            dilate_img = cv2.dilate(img, kernel, iterations=1)  # without this, some faces will have edges with wrong colors
            img = img * (1 - dilate_mask) + dilate_img * dilate_mask
            img = img.clip(0, 255) #.astype(np.uint8)
            atlas_img = torch.tensor(img/255.0).to(device).float()

        logger.info(f'unporject before optimize: {time.time() - start_unproject} s')

        if optimize_from is not None:
            start_optimzie = time.time()
            if optimize_from != 'None':
                # print('1. atlas_img.shape',atlas_img.shape) # [res,res,3]
                eye_positions = torch.tensor(eye_positions).float().to(device)
                up_dirs = torch.tensor(up_dirs).float().to(device)
                look_ats = torch.zeros((len(eye_positions), 3)).to(device)
                atlas_img = atlas_img.permute(2, 0, 1).flip(1)  # [3,res,res]

                if optimize_from == 'scratch':
                    init_atlas = None
                    shrinked_per_view_per_pixel_visibility = None
                elif optimize_from == 'naive': # naive:
                    init_atlas = atlas_img
                    shrinked_per_view_per_pixel_visibility = None
                elif optimize_from == 'ours':
                    init_atlas = atlas_img


                atlas_img, final_render_result = optimize_color(init_atlas, inpainted_images, vertices, faces, uvs,
                                                                mesh_tex_idx, cams, eye_positions, look_ats, up_dirs,
                                                                uv_centers, uv_scales, padding, inpaint_scale_factors,
                                                                glctx,
                                                                shrinked_per_view_per_pixel_visibility=
                                                                shrinked_per_view_per_pixel_visibility)  # [1,3,res,res], # [view_num,3,res,res]
                atlas_img = atlas_img[0].flip(1).permute(1, 2, 0)  # [res,res,3],
                for i in range(view_num):
                    save_CHW_RGB_img(final_render_result[i].detach().cpu().numpy(),
                                     os.path.join(root_path, 'meshes', cls_id, name, 'models',
                                                  f'final_render_result_{i}.png'))

                # print('2. atlas_img.shape', atlas_img.shape)
            logger.info(f' optimize: {time.time() - start_optimzie} s')
        '''render and see CD loss'''
      

        logger.info(f'unproject: {time.time() - start_unproject} s')
        # save mesh
        savemeshtes2(
            vertices.data.cpu().numpy(), # pointnp_px3
            uvs.data.cpu().numpy(), # tcoords_px2
            faces.data.cpu().numpy(), # facenp_fx3
            mesh_tex_idx.data.cpu().numpy(), # facetex_fx3

            os.path.join(root_path,'meshes',cls_id, name, 'models', 'model_normalized.obj') # fname
        )

        # save texture image

        # tex_map = atlas_img_visibility
        tex_map = atlas_img
        img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        dilate_mask = mask[0] # from [1,res,res,1] to [1,res,res]
        # dilate_mask = dilate_mask.detach().cpu().numpy()
        img *= dilate_mask.detach().cpu().numpy()  # added by PointDreamer author, necessary to enable later dialate
        img = img.clip(0, 255)
        dilate_mask = np.sum(img.astype(float), axis=-1,
                      keepdims=True)  # mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
        dilate_mask = (dilate_mask <= 3.0).astype(float)  # mask = (mask <= 3.0).astype(np.float)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)  # without this, some faces will have edges with wrong colors
        img = img * (1 - dilate_mask) + dilate_img * dilate_mask
        img = img.clip(0, 255).astype(np.uint8)

        print('img.shape',img.shape)
        PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
            os.path.join(root_path,'meshes',cls_id, name, 'models', 'model_normalized.png'))
        if debug:
            tex_map = atlas_img_view_id


            img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            dilate_mask = mask[0] # from [1,res,res,1] to [1,res,res]
            img *= dilate_mask.detach().cpu().numpy()  # added by PointDreamer author, necessary to enable later dialate
            img = img.clip(0, 255)
            dilate_mask = np.sum(img.astype(float), axis=-1,
                          keepdims=True)  # mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
            dilate_mask = (dilate_mask <= 3.0).astype(float)  # mask = (mask <= 3.0).astype(np.float)
            kernel = np.ones((3, 3), 'uint8')
            dilate_img = cv2.dilate(img, kernel, iterations=1)  # without this, some faces will have edges with wrong colors
            img = img * (1 - dilate_mask) + dilate_img * dilate_mask
            img = img.clip(0, 255).astype(np.uint8)

            PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                os.path.join(root_path,'meshes',cls_id, name, 'models', 'model_normalized_view_id.png'))


def optimize_color(atlas_img,inpainted_imgs,vertices,faces,uvs,mesh_tex_idx,
                   cams,eye_positions, look_ats, up_dirs,
                   uv_centers,uv_scales,padding,inpaint_scale_factors,glctx,
                   shrinked_per_view_per_pixel_visibility=None,
                   lr = 5e-2,iterations=100,print_every=10):
    '''

    :param atlas_img: [3,atlas_res,atlas_res]
    :param inpainted_imgs:  [view_num, 3,res,res]
    :param vertices: [num_vertices, 3]
    :param faces:[num_faces, 3]
    :param uvs: [num_uv_points, 2]
    :param mesh_tex_idx: [num_faces,3]: the three indexes of a face to index uvs
    :param uv_centers: [view_num,3]:
    :param uv_scales: [view_num]:
    :param scale_factors: [view_num]:
    :return:
    '''

    from kaolin.render.camera import perspective_camera, generate_perspective_projection,generate_transformation_matrix
    import math
    device = vertices.device
    # res = inpainted_imgs.shape[2]
    res = 1024

    # Prepare mesh data
    if atlas_img is not None:
        atlas_img = atlas_img.unsqueeze(0).detach().requires_grad_() # [1,3,,res,res]
    else:
        atlas_img = torch.rand((1, 3, 1024, 1024), dtype=torch.float, device=device, requires_grad=True)

    face_uvs = uvs[mesh_tex_idx].unsqueeze(0)  # [1,face_num,3,2]
    face_uvs = face_uvs.detach()
    face_uvs.requires_grad = False

    vertices = vertices.detach() # [v_num,3]
    vertices.requires_grad = False

    nb_faces = faces.shape[0]


    # Prepare training stuff
    optimizer = torch.optim.Adam([atlas_img], lr=lr)

    scheduler_step_size = 15
    scheduler_gamma = 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size,
                                                gamma=scheduler_gamma)
    MSE_loss_fn = torch.nn.MSELoss()

    # Prepare cameras
    num_views = len(cams)

    camera_fovy = fov = math.pi * 45 / 180
    cam_res = cams[0].height
    # cam_transform = []
    # for i in range(num_views):
    #     cam_transform.append(cams[i].extrinsics.view_matrix()[0,:,:3]) # [1,4,4]->[4,3]?
    # cam_transform = torch.stack(cam_transform,0) # [num_views,4,3]?
    cam_transforms = generate_transformation_matrix(eye_positions, look_ats, up_dirs).to(device)  # [8, 4, 3]

    cam_proj = generate_perspective_projection(camera_fovy,  ratio=1.0).to(device) # [3,1]?
    cam_proj = cam_proj.unsqueeze(0).repeat(num_views,1,1)
    # print('cam_transform.shape',cam_transform.shape)
    # print('cam_proj.shape',cam_proj.shape)


    '''
    vertices: [v_num,3]
    faces: [f_num,3]
    cam_proj: [B,3,1] # B = batch_size, can be view_num
    cam_transform: [B,4,3]
    face_uvs: [1,f_num,3,2]
    '''
    vertices = vertices.unsqueeze(0)
    vertices_batch = vertices
    face_vertices_camera, face_vertices_image, face_normals = \
        kal.render.mesh.prepare_vertices(
            vertices_batch.repeat(num_views, 1, 1),
            faces, cam_proj, camera_transform=cam_transforms
        )

    ### Perform Rasterization ###
    # Construct attributes that DIB-R rasterizer will interpolate.
    # the first is the UVS associated to each face
    # the second will make a hard segmentation mask
    face_attributes = [
        face_uvs.repeat(num_views, 1, 1, 1),
        torch.ones((num_views, nb_faces, 3, 1), device='cuda')
    ]
    use_nvdiffrast_rast = True
    if use_nvdiffrast_rast:
        pos = torch.zeros((len(cams), vertices.shape[1], 4), device=device)
        for i, cam in enumerate(cams):
            transformed_vertices = cam.transform(vertices)
            # Create a fake W (See nvdiffrast documentation)
            pos[i] = torch.nn.functional.pad(
                transformed_vertices, (0, 1), mode='constant', value=1.
            ).contiguous()

        vertice_uvs = pos[:, :, :2]
        # ori_vertice_uvs_min = vertice_uvs.min(1)[0]  # cam_num,2
        # ori_vertice_uvs_max = vertice_uvs.max(1)[0]  # cam_num,2

        # ori_vertice_uvs_min = ori_vertice_uvs_min.unsqueeze(1)  # cam_num,1,2
        # ori_vertice_uvs_max = ori_vertice_uvs_max.unsqueeze(1)  # cam_num,1,2
        # uv_centers = (ori_vertice_uvs_min + ori_vertice_uvs_max) / 2  # cam_num,1,2
        # uv_scales = (ori_vertice_uvs_max - ori_vertice_uvs_min).max(2)[0].unsqueeze(2)  # cam_num,1,2
        vertice_uvs = (vertice_uvs - uv_centers) / uv_scales  # now all between -0.5, 0.5
        vertice_uvs = vertice_uvs * (1 - 2 * padding)  # now all between -0.45, 0.45
        vertice_uvs = vertice_uvs * inpaint_scale_factors.unsqueeze(-1).unsqueeze(-1)
        vertice_uvs = vertice_uvs + 0.5  # now all between 0.05, 0.95
        vertice_uvs = vertice_uvs.clip(0, 1)


        pos[:, :, :2] = vertice_uvs * 2 - 1  # use the rescaled result to calculate masks, faceids and depths

        rast = dr.rasterize(glctx, pos, faces.int(), resolution=[res, res],  # cam.height, cam.width
                            grad_db=False)  # tuple of two tensors where we only use the first
        hard_mask = rast[0][:, :, :, -1:] != 0  # # [cam_num,res,res,1]
        face_idx = (rast[0][..., -1].long() - 1).contiguous()  # [cam_num,res,res]
        face_uvs_idx = mesh_tex_idx
        uv_map = nvdiffrast.torch.interpolate(uvs, rast[0], face_uvs_idx.int())[0]  # cam_num,res,res,2, right here

        mask = hard_mask
        soft_mask = hard_mask[..., 0]

        mask = torch.flip(mask, [1]) # [view_num,res,res,1]
        soft_mask = torch.flip(soft_mask, [1])
        uv_map = torch.flip(uv_map, [1])
        face_idx = torch.flip(face_idx, [1])

        texture_coords = uv_map
    else:  # don't use dibr_rasterization, seems somehow wrong
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            cam_res, cam_res, face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1])

        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features

    atlas_res = atlas_img.shape[3]
    texture_coords  # cam_num,res,res,2, ranging from (0,1)
    texture_coords_long = texture_coords * atlas_res
    texture_coords_long = texture_coords_long.long()
    texture_coords_long = torch.clip(texture_coords_long,0,atlas_res-1)

    for it in range(iterations):
        optimizer.zero_grad()

        images = kal.render.mesh.texture_mapping(texture_coords,
                                                atlas_img.repeat(num_views, 1, 1, 1),
                                                mode='bilinear') #[num_views,res,res,3]

        if shrinked_per_view_per_pixel_visibility is not None:


            indices =  texture_coords_long[:, :, :, 1],texture_coords_long[:, :, :, 0]

            shrinked_visibility_mask = shrinked_per_view_per_pixel_visibility[
                torch.arange(shrinked_per_view_per_pixel_visibility.size(0)).unsqueeze(-1).unsqueeze(-1),
                indices[0], indices[1]
            ] # # [cam_i, res, res]

            # shrinked_visibility_mask = shrinked_visibility_mask.unsqueeze(-1) #  [cam_i, res, res,1]


        images = torch.clamp(images * mask, 0., 1.)  # [num_views,H,W,C]
        images = torch.clamp(images, 0., 1.)  # [num_views,H,W,C]
        images = images.permute(0,3,1,2) # [num_views,C,H,W]



        foreground_masks = mask.permute(0,3,1,2).repeat(1,3,1,1) # [view_num,3,res,res]

        images *= foreground_masks.float()

        inpainted_imgs = transforms.Resize((res,res))(inpainted_imgs)
        inpainted_imgs *= foreground_masks.float()
        if shrinked_per_view_per_pixel_visibility is not None:
            shrinked_visibility_mask = shrinked_visibility_mask.unsqueeze(1) # #  [cam_i, res, res] to [cam_i,1, res, res]
            images *= shrinked_visibility_mask.float()
            inpainted_imgs *= shrinked_visibility_mask.float()
        vis=False
        if vis:
            for i in range(num_views):
                pred_img = images[i].detach().cpu().numpy()
                gt_img = inpainted_imgs[i].detach().cpu().numpy()
                cat = cat_images(pred_img,gt_img)
                display_CHW_RGB_img_np_matplotlib(cat)
        # per_pixel_MSE = MSE_loss_fn(image,inpainted_imgs)
        per_pixel_MSE = torch.mean(torch.abs(images - inpainted_imgs))

        loss = per_pixel_MSE

        loss.backward()
        optimizer.step()
        scheduler.step()

        # print loss
        # if (it) % print_every == 0 or it == (iterations - 1):
        #     print(it, '/', iterations, per_pixel_MSE.item())
    return atlas_img,images


def ours_main(render_after_inference, input_pc_generate_method, root_path, device, glctx,
              dataset_name,geo_root,logger,cls_id,coords_scale,noise_stddev, save_input_pc,smooth_mesh,
              eye_positions, up_dirs,
                refine_point_validation_by_remove_abnormal_depth,

                demo,
              **kwargs):

    split = 'test'
    if demo:
        split='demo'
    if dataset_name == 'shapenet_core_v2':
        from data.load_data_like_shapenet_core_v2 import ShapeNetCoreV2_Mesh_PC_Dataset
        dataset = ShapeNetCoreV2_Mesh_PC_Dataset(input_pc_generate_method=input_pc_generate_method,mesh_root=geo_root,
                                                 cls_id=cls_id, split=split, # test, demo
                                                 coords_scale=coords_scale, noise_stddev=noise_stddev,
                                                 smooth_mesh=smooth_mesh,
                                                 )

    elif dataset_name == 'google_scanned_objects':
        from data.load_data_like_shapenet_core_v2 import Google_Scanned_Objects_Mesh_PC_Dataset
        dataset = Google_Scanned_Objects_Mesh_PC_Dataset(mesh_root=geo_root,
                                                         coords_scale=coords_scale,
                                                         noise_stddev=noise_stddev)
        cls_id = dataset_name

    elif dataset_name == 'omniobject3d':
        from data.load_data_like_shapenet_core_v2 import Omniobject3d_Mesh_PC_Dataset
        dataset = Omniobject3d_Mesh_PC_Dataset(mesh_root=geo_root,coords_scale=coords_scale,noise_stddev=noise_stddev)
        cls_id = dataset_name

    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)



    _, cam_RTs = get_cam_Ks_RTs_from_locations(eye_positions)
    cam_RTs = torch.tensor(cam_RTs, device=device).float()


    refine_res = 512
    cam_K = np.array([
        [560.0, 0, 256, ],
        [0, 560, 256, ],
        [0, 0, 1, ],
    ])
    cam_K *= refine_res / 512
    cam_K[2, 2] = 1.0

    if not os.path.exists(os.path.join(root_path,'meshes')):
        os.makedirs(os.path.join(root_path,'meshes'))
    if not os.path.exists(os.path.join(root_path,'rendered_imgs')):
        os.makedirs(os.path.join(root_path,'rendered_imgs'))
    ''' inference: reconstruct textured mesh for each input shape '''
    # with torch.no_grad():

    logger.info(f'i\twhole_time\tname')
    all_shape_i = 0
    for batch_i, batch in enumerate(dataloader):
        try:

            coords_batch, colors_batch, vertices_batch, faces_batch, f_normals_batch,\
            unsmoothed_vertices_batch,unsmoothed_faces_batch, names_batch = batch

            names = names_batch
            for shape_id in range(len(names_batch)):
                start_sample = time.time()
                all_shape_i += 1

                logger.info('\n------------------------------------')
                logger.info(f'{all_shape_i}/{len(dataset)}\t{names_batch[shape_id]}')
                # if all_shape_i<132: # debug
                #     continue
                print(all_shape_i, '/', len(dataset), names_batch[shape_id])
                coords = coords_batch[shape_id].to(device)
                colors = colors_batch[shape_id].to(device)

                # reduce point num here: # debug
                reduce_point_num=False
                reduced_point_num = 20000
                if reduce_point_num:
                    coords = coords[:reduced_point_num]
                    colors = colors[:reduced_point_num]


                if save_input_pc:

                    save_pc_path = os.path.join(root_path, 'meshes',cls_id,names[shape_id], 'models', f'input_pc.ply')
                    os.makedirs(os.path.dirname(save_pc_path),exist_ok=True)
                    save_colored_pc_ply(coords.detach().cpu().numpy() / coords_scale,
                                        colors.detach().cpu().numpy(), save_pc_path)  # save the none-scaled coords

                vertices = vertices_batch[shape_id].to(device)
                faces = faces_batch[shape_id].to(device)
                f_normals = f_normals_batch[shape_id].to(device)
                unsmoothed_vertices = unsmoothed_vertices_batch[shape_id].to(device)
                unsmoothed_faces = unsmoothed_faces_batch[shape_id].to(device)
                # vis_input = True
                # if vis_input:
                #     from utils.vtk_basic import vis_actors_vtk,get_mesh_actor_vtk,get_pc_actor_vtk
                #     # mesh_trimesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(),
                #     #                                faces=faces.detach().cpu().numpy())
                #     vis_actors_vtk([
                #         # get_mesh_actor_vtk(mesh_trimesh),
                #         get_pc_actor_vtk(np.array([[0,0,0]])),
                #     ])

                recon_one_shape(name=names_batch[shape_id],cls_id=cls_id,root_path=root_path,device=device,glctx=glctx,
                                coords=coords, colors=colors, vertices=vertices, faces=faces, f_normals=f_normals,
                                unsmoothed_vertices=unsmoothed_vertices, unsmoothed_faces=unsmoothed_faces,
                                eye_positions = eye_positions,cam_RTs=cam_RTs,cam_K=cam_K,refine_res=refine_res,
                                refine_point_validation_by_remove_abnormal_depth=refine_point_validation_by_remove_abnormal_depth,
                                up_dirs=up_dirs,logger=logger,
                                **kwargs)
                end = time.time()
                duration = end-start_sample
                logger.info(f'whole_time: {duration} s')

        except KeyboardInterrupt:
            logger.error((traceback.format_exc()))
            sys.exit(0)
        except:
            # logger.error(f'ERROR: current id: {i}')
            logger.error(f'ERROR')
            logger.error(f'cls_id: {cls_id}')
            logger.error(f'cls_id: {cls_id}')


            # if os.path.exists(shape_dir):
            #     shutil.rmtree(shape_dir)
            #     logger.error(f'delete file:{shape_dir}')
            # else:
            #     logger.error(f"didn't delete file since it doesn't exist: {shape_dir}")

            logger.error(traceback.format_exc())
    ''' render textured mesh'''
    if render_after_inference:
        current_dir = os.getcwd()
        if input_pc_generate_method == 'blender':
            path = os.path.abspath(root_path)
            os.chdir('depth_renderer/')
            command = 'python run_render_albedo.py --shapenet_normalized_path ' + path

            os.system(command)

            os.system(command)

        elif input_pc_generate_method == 'kaolin':
            print('kaolin !!!')


            # save_root_path = render_path = mesh_root_path.replace('meshes','rendered_imgs')
            save_root_path = render_path = root_path
            render_textured_meshes_shapenet2(names=dataset.names, root_path=root_path, device=device,
                                             save_root_path=save_root_path,glctx=glctx)
        evaluate_after_render = True
        if evaluate_after_render:
            os.chdir(current_dir)
            # command = f'python run_evaluation.py --dataset_name={dataset_name} --pred_root_path={root_path}/rendered_imgs'
            # os.system(command)
            from run_evaluation import eval
            eval(dataset_name,pred_root_path=f'{root_path}/rendered_imgs')
