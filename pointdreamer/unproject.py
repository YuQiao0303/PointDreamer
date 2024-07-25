import os

import torch
import numpy as np
import cv2
import math
from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images,save_CHW_RGB_img,detect_edges_in_gray_by_scharr
from utils.utils_2d import detect_edges_in_gray_by_scharr_torch_batch,dilate_torch_batch
from utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
from pointdreamer.ours_utils import get_point_validation_by_depth,naive_inpainting
import kaolin as kal
import time
from utils.mesh_utils import subdivide_with_uv


## Deal with invisible areas
def compute_vertex_only_uv_mask(face_vertex_idx, face_uvs_idx):
    device = face_vertex_idx.device
    # Step 1: Flatten the face_vertex_idx and face_uvs_idx arrays
    face_vertex_idx_flat = face_vertex_idx.flatten()
    face_uvs_idx_flat = face_uvs_idx.flatten()

    # Step 2: Create counts for each vertex and uv index
    V = face_vertex_idx.max().item() + 1  # number of vertices
    vertex_uv_counts = torch.zeros(V, dtype=torch.int64).to(device)

    # Step 3: Count the occurrences of vertex-uv pairs
    vertex_uv_pairs = torch.stack((face_vertex_idx_flat, face_uvs_idx_flat), dim=1)
    unique_vertex_uv_pairs, counts = torch.unique(vertex_uv_pairs, dim=0, return_counts=True)

    # Step 4: Update the vertex_uv_counts based on unique vertex-uv pairs
    vertex_uv_counts.scatter_add_(0, unique_vertex_uv_pairs[:, 0], torch.ones_like(counts).to(device))

    # Step 5: Determine if each vertex has only one corresponding uv index
    vertex_only_uv_mask = vertex_uv_counts == 1

    return vertex_only_uv_mask,unique_vertex_uv_pairs

def paint_invisible_areas_by_optimize(atlas_img,points,points_atlas_pixel_coord,point_view_ids,input_xyz,input_rgb,
                                      lr=1e-2,iterations = 400,debug=False,print_step=10):
    '''
    input_rgb: [B, N,3]. from 0-1
    '''
    
    # vis_actors_vtk([])
    device = input_xyz.device
    from models.TextureField.TF_Network import Network
    model = Network(decoder_only=True,  device=device).train().requires_grad_(True).to(device)  
    # prepare 
    B = 1
    xz = torch.randn([B, 32, 64, 64], device=device).requires_grad_()
    xy = torch.randn([B, 32, 64, 64], device=device).requires_grad_()
    yz = torch.randn([B, 32, 64, 64], device=device).requires_grad_()

    # optimizer = torch.optim.Adam([xz, xy, yz,model.parameters()], lr=lr)
    params_to_optimize = list(model.parameters()) + [xz, xy, yz]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
  
    MSE_loss_fn = torch.nn.MSELoss()

    input_rgb_2 =  input_rgb.unsqueeze(0)*2-1 # # 1,N,3, ranging from -1 to 1

    # optimization loop
    for it in range(iterations):
        optimizer.zero_grad()
        rgb_feature = {'xz': xz,  'xy':xy,  'yz': yz}
        pred_rgb = model.pred_rgb(rgb_feature, input_xyz.unsqueeze(0)) # 1,N,3
        loss = MSE_loss_fn(pred_rgb,input_rgb_2)

        loss.backward()
        optimizer.step()
        
        if debug and it % print_step == 0:
            print( it, ': loss', loss.item())
    
    # predict result
    invisible_point_mask = point_view_ids<0
    pred_rgb = model.pred_rgb(rgb_feature, points[invisible_point_mask].unsqueeze(0))[0] # ,N,3
    atlas_img[points_atlas_pixel_coord[invisible_point_mask][:, 0],
                        points_atlas_pixel_coord[invisible_point_mask][:, 1]] = (pred_rgb*0.5+0.5).clip(0,1) # # [N,3]
    
    # import kiui
    # kiui.lo(temp)
    # from utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
    # # vis_actors_vtk([])
    # vis_actors_vtk([
    #     get_colorful_pc_actor_vtk(input_xyz.detach().cpu().numpy(),input_rgb.detach().cpu().numpy()),
    #     get_colorful_pc_actor_vtk(input_xyz.detach().cpu().numpy()+1,(pred_rgb[0]*0.5+0.5).clip(0,1).detach().cpu().numpy()),
    # ])

    return atlas_img

def paint_invisible_areas_by_neighbors(vertices,faces,uvs,face_uv_idx,to_inpaint_face_id,atlas_img, atlas_inpainted_mask,use_atlas=True) :
    """
    Modified from: https://github.com/AiuniAI/Unique3D/blob/main/scripts/project_mesh.py: complete_unseen_vertex_color
    meshes: the mesh with vertex color to be completed.
    valid_index: the index of the valid vertices, where valid means colors are fixed. [V, 1]
    """
    V = len(vertices)
    device = vertices.device
    atlas_res = atlas_inpainted_mask.shape[1] # [res,res]
    per_pixel_mask = atlas_inpainted_mask#[0,:,:,0]
    
    
    ## Subdivide the mesh, so that even with per vertex color, the color resolution wouldn't be too low
    subdivided_vertices = vertices.cpu().numpy()
    subdivided_faces = faces.cpu().numpy()
    subdivided_uvs = uvs.cpu().numpy()
    subdivided_face_uv_idx = face_uv_idx.cpu().numpy()
    to_inpaint_face_id = to_inpaint_face_id.cpu().numpy()
    iterations = 2
    for i in range(iterations):
        subdivided_vertices, subdivided_faces, subdivided_uvs, subdivided_face_uv_idx = subdivide_with_uv(
            subdivided_vertices, subdivided_faces, subdivided_face_uv_idx,subdivided_uvs,face_index=to_inpaint_face_id)
    
    subdivided_vertices = torch.tensor(subdivided_vertices).to(device)
    subdivided_faces = torch.tensor(subdivided_faces).to(device).long()
    subdivided_uvs = torch.tensor(subdivided_uvs).to(device)
    subdivided_face_uv_idx = torch.tensor(subdivided_face_uv_idx).to(device).long()
    
    
    ## Get per vertex color of the subdivided mesh
    vertex_only_uv_mask,unique_vertex_uv_pairs = compute_vertex_only_uv_mask(subdivided_faces,subdivided_face_uv_idx)
    subdivided_vert_uvs = torch.zeros(len(subdivided_vertices), 2).to(device)

    unique_vertex_uvs = subdivided_uvs[unique_vertex_uv_pairs[:, 1]]
    subdivided_vert_uvs[unique_vertex_uv_pairs[:, 0]] = unique_vertex_uvs

    
    subdivided_vert_uv_pixel_coords = (subdivided_vert_uvs*atlas_res).clip(0,atlas_res-1).long().to(device)
    subdivided_vert_uv_pixel_coords = torch.cat((subdivided_vert_uv_pixel_coords[..., 1].unsqueeze(-1),
                                        subdivided_vert_uv_pixel_coords[..., 0].unsqueeze(-1)),
                            dim=-1)  # switch x and y if you ever need to query pixel coordiantes
    subdevided_vert_colors = atlas_img[subdivided_vert_uv_pixel_coords[:, 0], subdivided_vert_uv_pixel_coords[:, 1]]
    
    
    subdevided_vert_has_color = per_pixel_mask[subdivided_vert_uv_pixel_coords[:, 0], subdivided_vert_uv_pixel_coords[:, 1]]
    valid_index = torch.arange(len(subdivided_vertices))[subdevided_vert_has_color]


    # Inpaint no_color vertices by neighbors (modified from Unique3D)
    colors = subdevided_vert_colors
    V = colors.shape[0]
    
    invalid_index = torch.ones_like(colors[:, 0]).bool()    # [V]
    invalid_index[valid_index] = False
    invalid_index = torch.arange(V).to(device)[invalid_index]

    L = kal.ops.mesh.uniform_laplacian(V, subdivided_faces)
    E = torch.sparse_coo_tensor(torch.tensor([list(range(V))] * 2), torch.ones((V,)), size=(V, V)).to(device) # eye
    L = L + E

    colored_count = torch.ones_like(colors[:, 0])   # [V]
    colored_count[invalid_index] = 0
    L_invalid = torch.index_select(L, 0, invalid_index)    # sparse [IV, V]
    
    total_colored = colored_count.sum()
    coloring_round = 0
    stage = "uncolored"
    from tqdm import tqdm
    pbar = tqdm(miniters=100)
    while stage == "uncolored" or coloring_round > 0:
        new_color = torch.matmul(L_invalid, colors * colored_count[:, None])    # [IV, 3]
        new_count = torch.matmul(L_invalid, colored_count)[:, None]             # [IV, 1]
        colors[invalid_index] = torch.where(new_count > 0, new_color / new_count, colors[invalid_index])
        colored_count[invalid_index] = (new_count[:, 0] > 0).float()
        
        new_total_colored = colored_count.sum()
        if new_total_colored > total_colored:
            total_colored = new_total_colored
            coloring_round += 1
        else:
            stage = "colored"
            coloring_round -= 1
        pbar.update(1)
        if coloring_round > 10000:
            print("coloring_round > 10000, break")
            break
    assert not torch.isnan(colors).any()
    subdevided_vert_colors = colors
    
    
    ### Update atlas img
    if use_atlas:
        subdevided_vert_colors
        subdivided_vert_uv_pixel_coords
        atlas_img[subdivided_vert_uv_pixel_coords[:, 0], subdivided_vert_uv_pixel_coords[:, 1]] = subdevided_vert_colors
        atlas_inpainted_mask[subdivided_vert_uv_pixel_coords[:, 0], subdivided_vert_uv_pixel_coords[:, 1]] =  True
        
       
        atlas_img = naive_inpainting(img=atlas_img.permute(2, 0, 1), 
                                     no_need_inpaint_mask2=atlas_inpainted_mask.unsqueeze(0),method='nearest')
        atlas_img = torch.tensor(atlas_img, device=device).permute(1, 2, 0)
        return atlas_img
    else:
        return subdivided_vertices,subdivided_faces,subdevided_vert_colors



## Unproject and Non-Border-First
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


        similarity_between_point_normal_and_view_dir = per_point_face_normal @ base_dirs.t()  # [ point_num,view_num]

        # Get per view per point pixel (for each point, its corresponding pixel coordnate in each view image)
        per_view_per_point_pixel = per_view_per_point_uvs * view_img_res
        per_view_per_point_pixel = per_view_per_point_pixel.clip(0, view_img_res - 1)
        per_view_per_point_pixel = per_view_per_point_pixel.long()
        per_view_per_point_pixel = torch.cat((per_view_per_point_pixel[:, :, 1].unsqueeze(-1),
                                                per_view_per_point_pixel[:, :, 0].unsqueeze(-1)),
                                    dim=-1)  # switch x and y if you ever need to query pixel coordiantes



        # per_point_view_weight =  similarity_between_point_normal_and_view_dir
        # per_point_view_weight[~(per_view_per_point_visibility.permute(1,0).bool())] -=100
        '''Non-Border-First Unprojection (UBF)'''
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


        # # if a point is not visible in any view's non-border area, now we consider all areas, no matter border or not, by using the unshrinked visibility
        # per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)
        # candidate_per_point_per_view_mask[per_point_left_view_num < 1, :] = \
        #     torch.logical_or(
        #         candidate_per_point_per_view_mask[per_point_left_view_num < 1, :],
        #         per_view_per_point_visibility.permute(1, 0)[per_point_left_view_num < 1, :]
        #     )


        # now choose the ones with best normal similarity
        per_point_per_view_weight = torch.softmax(similarity_between_point_normal_and_view_dir,1) # [pointnum, view_num]
        per_point_per_view_weight[~candidate_per_point_per_view_mask] = -100
        point_view_ids = torch.argmax(per_point_per_view_weight, dim=1)
        point_view_ids[candidate_per_point_per_view_mask.sum(1)<1] = -100 #view_num # if has no visible view, make it black
        # per_point_left_view_num = candidate_per_point_per_view_mask.sum(1)
        # candidate_per_point_per_view_mask[per_point_left_view_num > 1] = \
        #     candidate_per_point_per_view_mask[
        #         per_point_left_view_num > 1] * ~best_normal_per_point_per_view_mask[per_point_left_view_num>1]
        # # get the final result
        # point_view_ids = torch.argmax(candidate_per_point_per_view_mask.long(), dim=1)



        single_view_atlas_imgs = torch.zeros((view_num,res, res, 3), device=device)
        single_view_atlas_masks = torch.zeros((view_num,res, res, 3), device=device).bool()
        atlas_img = torch.zeros((res, res, 3), device=device)
        atlas_painted_mask = torch.zeros((res,res),device=device).bool()
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

            atlas_painted_mask[points_atlas_pixel_coord[point_this_view_mask][:, 0],
                        points_atlas_pixel_coord[point_this_view_mask][:, 1]] = True

            ################
            # # save single_view_atlas_imgs
            # per_view_per_point_visibility
            # shrinked_per_view_per_point_visibility
            # all_visible_point_this_view_mask = shrinked_per_view_per_point_visibility.bool()[i]
            # single_view_atlas_imgs[i][points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 0],
            #             points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 1]] = \
            #     view_img[per_view_per_point_pixel[i][all_visible_point_this_view_mask][:,0],
            #                 per_view_per_point_pixel[i][all_visible_point_this_view_mask][:,1]]


            # single_view_atlas_masks[i][points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 0],
            #                             points_atlas_pixel_coord[all_visible_point_this_view_mask][:, 1]] = True
            # # single_view_atlas_imgs[i][~single_view_atlas_masks[i][..., 0]] = palette[i]

            # # save_CHW_RGB_img(single_view_atlas_imgs[i].permute(2,0,1).detach().cpu().numpy()[:, ::-1, :],
            # #                     os.path.join(root_path,'meshes',cls_id, name, 'models',f'atlax_view_{i}.png'))

        # now deal with invisible areas


      
        
        return atlas_img,shrinked_per_view_per_pixel_visibility,point_view_ids,points_atlas_pixel_coord,points,atlas_painted_mask
    


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
                                                 (~per_view_atlas_border_mask)) # [view_num,res,res]
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



## Dilate atlas to avoid artifact at edges
def dilate_atlas(atlas_img,mask):
    # without this, face edges would look weird
    tex_map = atlas_img
    lo, hi = (0, 1)
    img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    dilate_mask = mask[0]  # from [1,res,res,1] to [1,res,res]
    res = mask.shape[1]
    device = atlas_img.device
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
    return atlas_img


