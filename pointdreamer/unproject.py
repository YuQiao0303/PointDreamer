import os.path

import torch
import numpy as np
import cv2
import math
from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images,save_CHW_RGB_img,detect_edges_in_gray_by_scharr
from utils.utils_2d import detect_edges_in_gray_by_scharr_torch_batch,dilate_torch_batch
from pointdreamer.ours_utils import get_point_validation_by_depth
import time

def unproject(inpainted_images,vertices,f_normals,
              view_img_res,
              cams,cam_res,base_dirs,
              gb_pos,mask,per_atlas_pixel_face_id,
              uv_centers,uv_scales,padding,inpaint_scale_factors,
              mesh_normalized_depths,edge_dilate_kernels,save_img_path):
    with torch.no_grad():
        '''
        uvs:                per mesh vertex uv:                     [vert_num, 2]
        mesh_tex_idx:       per face uv coord id (to index uvs):    [face_num,3]
        gb_pos:             per pixel 3D coordinate:                [1,res,res,3] 
        mask:               per pixel validation:                   [1,res,res,1]
        per_pixel_face_id:  per pixel face id:                      [1,res,res]
        
        '''

        # view_img_res = res
        # res = xatlas_texture_res
        res = mask.shape[1]
        view_num = len(cams)
        device = vertices.device

        per_pixel_mask = mask[0,:,:,0] # [res,res]
        per_pixel_point_coord = gb_pos[0] # [res,res,3]
        per_atlas_pixel_face_id = per_atlas_pixel_face_id[0] #[1,res,res]

        per_pixel_pixel_coord = torch.zeros((res,res,2),device=device).long()
        xx, yy = torch.meshgrid(torch.arange(res).to(device), torch.arange(res).to(device))
        per_pixel_pixel_coord[:, :, 0] = xx
        per_pixel_pixel_coord[:, :, 1] = yy

        points = per_pixel_point_coord[per_pixel_mask] # [?,3] ??
        points_atlas_pixel_coord = per_pixel_pixel_coord[per_pixel_mask].long() # [?,2] ??


        # get per-atlas-pixel's corresponding depth and uv in multiview images 
        # (depth used for calculating visibility, uv used for query correspondign color)

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


        # Get per-atls-pixel  visibility by depth (so that we have per-view visible atlas)
        per_view_per_point_visibility,_ = get_point_validation_by_depth(cam_res,per_view_per_point_uvs_no_scale,
                                        per_view_per_point_depths,mesh_normalized_depths,offset = 0.0001,
                                                                        vis=False)# [cam_num, point_num]

        start_shrink_visibility = time.time()
     


        per_atlas_pixel_per_view_visibility = torch.zeros((res,res,view_num),device=device).bool()

        per_atlas_pixel_per_view_visibility[per_pixel_mask] = per_view_per_point_visibility.permute(1,0)#.clone() # (res,res,view_num)

        # shrink per-view visible atlas (remove border areas, only keep non-border areas for later use)
        per_kernel_per_view_shrinked_per_pixel_visibility = get_shrinked_per_view_per_pixel_visibility_torch(
            per_pixel_mask,per_atlas_pixel_per_view_visibility,
            kernel_sizes= edge_dilate_kernels*(res//256),
            save_path = os.path.join(save_img_path,'shrink_per_view_edge')) # [kernel_num,view_num,res,res]
     
        # try:
        #     logger.info(f'shrink visibility: {time.time() - start_shrink_visibility} s')
        # except:
        #     pass

        # Get direction priority (similarity between point normal and view_dir)
        per_atlas_pixel_face_normal = f_normals[per_atlas_pixel_face_id] #res,res,3
        # print('f_normals.shape',f_normals.shape) #[face_num,3]
        # print('per_atlas_pixel_face_id.shape',per_atlas_pixel_face_id.shape) #[res,res]
        # print('per_atlas_pixel_face_normal.shape', per_atlas_pixel_face_normal.shape)
        per_point_face_normal = per_atlas_pixel_face_normal[per_pixel_mask] # [?,3]


        similarity_between_point_normal_and_view_dir = per_point_face_normal @ torch.tensor(base_dirs,
                                                                                            device=device).t()  # [ point_num,view_num]

        # Get per view per point pixel (for each point, its corresponding pixel coordnate in each view image)
        per_view_per_point_pixel = per_view_per_point_uvs * view_img_res
        per_view_per_point_pixel = per_view_per_point_pixel.clip(0, view_img_res - 1)
        per_view_per_point_pixel = per_view_per_point_pixel.long()
        per_view_per_point_pixel = torch.cat((per_view_per_point_pixel[:, :, 1].unsqueeze(-1),
                                                per_view_per_point_pixel[:, :, 0].unsqueeze(-1)),
                                    dim=-1)  # switch x and y if you ever need to query pixel coordiantes



        # per_point_view_weight =  similarity_between_point_normal_and_view_dir
        # per_point_view_weight[~(per_view_per_point_visibility.permute(1,0).bool())] -=100
        '''24.01.05: Non-Border-First Unprojection (UBF)'''
        point_num = per_point_face_normal.shape[0]
        # candidate_per_point_per_view_mask = torch.ones((point_num,view_num)).bool().to(device) # [point_num,view_num]

        # first use shrinked visibility (only contains non-border areas) 
        shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[0]
        shrinked_per_view_per_point_visibility = \
            shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

        candidate_per_point_per_view_mask = \
            shrinked_per_view_per_point_visibility.permute(1, 0)  # [point_num,view_num]

        # multi-level NBF: the size of border areas can be controled by the dilation kernels
        for i in range(1,len(edge_dilate_kernels)):
            # if a point is not visible in any view, try less tight mask 
            # (we have multiple shrinked visibility mask with different dilation kernels. 
            # a smaller kernel means a smaller area is regarded as border area, which enables more areas to be considered by projection (less tight mask)
            per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)

            shrinked_per_view_per_pixel_visibility = per_kernel_per_view_shrinked_per_pixel_visibility[i]
            shrinked_per_view_per_point_visibility = \
                shrinked_per_view_per_pixel_visibility.permute(1, 2, 0)[per_pixel_mask].permute(1, 0)

            candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
                torch.logical_or(
                    candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
                    shrinked_per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
                )


        # if a point is not visible in any view's non-border area, now we consider all areas, no matter border or not, by using the unshrinked visibility
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
        # point_view_ids[candidate_per_point_per_view_mask.sum(1)<1] = view_num # if has no visible view, make it black
        # per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)
        # candidate_per_point_per_view_mask[per_point_left_view_num > 1] = \
        #     candidate_per_point_per_view_mask[
        #         per_point_left_view_num > 1] * ~best_normal_per_point_per_view_mask[per_point_left_view_num>1]
        # # get the final result
        # point_view_ids = torch.argmax(candidate_per_point_per_view_mask.long(), dim=1)



        single_view_atlas_imgs = torch.zeros((view_num,res, res, 3), device=device)
        single_view_atlas_masks = torch.zeros((view_num,res, res, 3), device=device).bool()
        atlas_img = torch.zeros((res, res, 3), device=device)
        per_pixel_view_id = -torch.ones((res,res),device=device).long()



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
            # single_view_atlas_imgs[i][~single_view_atlas_masks[i][..., 0]] = palette[i]

            # save_CHW_RGB_img(single_view_atlas_imgs[i].permute(2,0,1).detach().cpu().numpy()[:, ::-1, :],
            #                     os.path.join(root_path,'meshes',cls_id, name, 'models',f'atlax_view_{i}.png'))

      
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
        kernel *= res // 256
        dilate_img = cv2.dilate(img, kernel, iterations=1)  # without this, some faces will have edges with wrong colors
        img = img * (1 - dilate_mask) + dilate_img * dilate_mask
        img = img.clip(0, 255) #.astype(np.uint8)
        atlas_img = torch.tensor(img/255.0).to(device).float()
        return atlas_img,shrinked_per_view_per_pixel_visibility
    



def get_shrinked_per_view_per_pixel_visibility_torch(per_pixel_mask,per_atlas_pixel_per_view_visibility,
                                                     kernel_sizes = [21],save_path=None):
    '''

    :param per_pixel_mask: res,res
    :param per_atlas_pixel_per_view_visibility:
    :return:
    '''
    if kernel_sizes[0] == 0:
        return per_atlas_pixel_per_view_visibility.permute(2,0,1).unsqueeze(0)
    device = per_atlas_pixel_per_view_visibility.device
    view_num = per_atlas_pixel_per_view_visibility.shape[-1]
    atlas_background_edges = detect_edges_in_gray_by_scharr_torch_batch(
        per_pixel_mask.unsqueeze(0).unsqueeze(0).float() * 255.0)  # [1,1,res,res]
    atlas_background_edges_mask = atlas_background_edges > 125  # [1,1,res,res]
    atlas_background_edges_mask = atlas_background_edges_mask[0] # [1,res,res]

    per_view_atlas_edges = detect_edges_in_gray_by_scharr_torch_batch(
        per_atlas_pixel_per_view_visibility.permute(2, 0, 1).unsqueeze(1).float() * 255.0)  # [view_num,1,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges > 255.0 / 2 - 1  # [view_num,1,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges_mask.squeeze(1)  # [view_num,res,res]
    per_view_atlas_edges_mask = per_view_atlas_edges_mask * ~atlas_background_edges_mask # [view_num,res,res]
    per_kernel_per_view_shrinked_per_pixel_visibility = []
    for kernel_size in kernel_sizes:
        per_view_atlas_border_mask = dilate_torch_batch(per_view_atlas_edges_mask.float() * 255.0,
                                                        kernel_size=kernel_size)  # [view_num,res,res]
        per_view_atlas_border_mask = per_view_atlas_border_mask>255.0/2
        shrinked_per_view_per_pixel_visibility = (per_atlas_pixel_per_view_visibility.permute(2, 0, 1) * \
                                                  torch.tensor(~per_view_atlas_border_mask).bool().to(device)) # [view_num,res,res]
        per_kernel_per_view_shrinked_per_pixel_visibility.append(shrinked_per_view_per_pixel_visibility)
    per_kernel_per_view_shrinked_per_pixel_visibility = torch.stack(per_kernel_per_view_shrinked_per_pixel_visibility,0)  # [kernel_num,view_num,res,res]
    if save_path is not None:
        os.makedirs(save_path,exist_ok=True)


        src_img_color_with_edges = per_atlas_pixel_per_view_visibility.permute(2,0,1).clone().float().unsqueeze(-1).repeat(1,1,1,3)  # [view_num,res,res,3,]

        src_img_color_with_edges[atlas_background_edges_mask.repeat(view_num,1,1)]=  torch.tensor([[1.0,0,0]],device=device) # background edges painted in red
        src_img_color_with_edges[per_view_atlas_edges_mask] = torch.tensor([[0,0,1.0]],device=device)


        src_img_color_with_edges = src_img_color_with_edges.permute(0,3,1,2)  # [view_num,3,res,res,]
        for i in range(view_num):
            cat = cat_images(src_img_color_with_edges[i].detach().cpu().numpy(),
                             per_view_atlas_edges_mask[i].float().unsqueeze(0).repeat(3,1,1).detach().cpu().numpy())
            cat = cat_images(cat,per_view_atlas_border_mask[i].float().unsqueeze(0).repeat(3,1,1).detach().cpu().numpy())
            save_CHW_RGB_img(cat[:,::-1,:],os.path.join(save_path,f'{i}.png'))
    return per_kernel_per_view_shrinked_per_pixel_visibility


