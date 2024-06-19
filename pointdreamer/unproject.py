import os.path

import torch
import numpy as np
import cv2
import math
from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images,save_CHW_RGB_img,detect_edges_in_gray_by_scharr
from utils.utils_2d import detect_edges_in_gray_by_scharr_torch_batch,dilate_torch_batch



# def get_border_area_in_atlax_view_id_torch_batch(per_pixel_view_id,background_edges_mask,dilate_kernel_size =11,save_path=None):
#     '''

#     :param per_pixel_view_id: [res,res], -1 means background
#     :param atlas_img_view_id: [3,res,res], black means background
#     :return:
#     '''
#     # if per_pixel_view_id.max()-1 > view_num:
#     #     view_num = per_pixel_view_id.max()-1
#     # img = per_pixel_view_id+1
#     # res = img.shape[0]
#     #
#     #
#     #
#     # uint8_img = img * math.floor((255.0/(view_num-1)))
#     # # print('uint8_img',uint8_img.min(),uint8_img.max(),uint8_img)
#     # uint8_img = np.clip(uint8_img, 0, 255)
#     # uint8_img = uint8_img.astype(np.uint8)
#     #
#     # # detect edges:
#     # im1x = cv2.Scharr(uint8_img, cv2.CV_64F, 1, 0)
#     # im1y = cv2.Scharr(uint8_img, cv2.CV_64F, 0, 1)
#     # im1x = cv2.convertScaleAbs(im1x)
#     # im1y = cv2.convertScaleAbs(im1y)
#     # edges = cv2.addWeighted(im1x, 0.5, im1y, 0.5, 0)
#     #
#     #
#     #
#     #
#     # edge_thresh = math.floor((255.0/(view_num-1))) -1
#     # edge_mask = edges > edge_thresh # res,res
#     #
#     # # vis = False
#     # # if vis:
#     # #     edges_binary = edges.copy()
#     # #     edges_binary[edges > edge_thresh] = 0
#     # #     edges_binary[edges <= edge_thresh] = 255 # res,res
#     # #     src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)  # [H,W,3]
#     # #     src_img_color_with_edges = src_img_color.copy() # [H,W,3]
#     # #     src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  # [H,W,3]
#     # #     display_CHW_RGB_img_np_matplotlib(src_img_color_with_edges.transpose(2, 0, 1))

#     # if per_pixel_view_id.max() + 1 > view_num:
#     #     view_num = per_pixel_view_id.max() + 1
#     view_num = per_pixel_view_id.max() + 1

#     uint8_img = per_pixel_view_id * math.floor((255.0 / (view_num - 1)))
#     uint8_img = torch.clip(uint8_img, 0, 255)#.astype(np.uint8)

#     ### Get edges


#     edge_thresh = math.floor((255.0 / (view_num))) - 1
#     edges = detect_edges_in_gray_by_scharr(uint8_img)
#     edge_mask = edges > edge_thresh  # res,res
#     edge_mask = edge_mask * ~background_edges_mask


#     # get border area by dilating edges
#     dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
#     dilate_iterations = 1
#     border_mask = cv2.dilate(edge_mask.astype(np.uint8)*255, dilate_kernel,
#                                          iterations=dilate_iterations).astype(np.bool_) # [res,res]


#     if save_path is not None:
#         os.makedirs(os.path.dirname(save_path),exist_ok=True)
#         edges_binary = edges.copy()
#         edges_binary[edges > edge_thresh] = 0
#         edges_binary[edges <= edge_thresh] = 255  # res,res

#         src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)  # [H,W,3]
#         src_img_color_with_edges = src_img_color.copy()  # [H,W,3]
#         src_img_color_with_edges[background_edges_mask] = (255, 0, 0)  # [H,W,3]
#         src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  # [H,W,3]

#         cat = cat_images(src_img_color_with_edges.transpose(2, 0, 1), edge_mask[np.newaxis, ...])
#         cat = cat_images(cat,np.repeat((border_mask.astype(np.float)/1.0)[np.newaxis, ...],3,0))
#         save_CHW_RGB_img(cat[:,::-1,:],save_path)

#     return edge_mask,border_mask # both [res,res]


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


# def get_border_area_in_atlax_img(atlas_img,background_edges_mask,edge_thresh=125,dilate_kernel_size =11):

#     # if per_pixel_view_id.max()-1 > view_num:
#     #     view_num = per_pixel_view_id.max()-1
#     # img = per_pixel_view_id+1
#     # res = img.shape[0]
#     #
#     #
#     #
#     # uint8_img = img * math.floor((255.0/(view_num-1)))
#     # # print('uint8_img',uint8_img.min(),uint8_img.max(),uint8_img)
#     # uint8_img = np.clip(uint8_img, 0, 255)
#     # uint8_img = uint8_img.astype(np.uint8)
#     #
#     # # detect edges:
#     # im1x = cv2.Scharr(uint8_img, cv2.CV_64F, 1, 0)
#     # im1y = cv2.Scharr(uint8_img, cv2.CV_64F, 0, 1)
#     # im1x = cv2.convertScaleAbs(im1x)
#     # im1y = cv2.convertScaleAbs(im1y)
#     # edges = cv2.addWeighted(im1x, 0.5, im1y, 0.5, 0)
#     #
#     #
#     #
#     #
#     # edge_thresh = math.floor((255.0/(view_num-1))) -1
#     # edge_mask = edges > edge_thresh # res,res
#     #
#     # # vis = False
#     # # if vis:
#     # #     edges_binary = edges.copy()
#     # #     edges_binary[edges > edge_thresh] = 0
#     # #     edges_binary[edges <= edge_thresh] = 255 # res,res
#     # #     src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)  # [H,W,3]
#     # #     src_img_color_with_edges = src_img_color.copy() # [H,W,3]
#     # #     src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  # [H,W,3]
#     # #     display_CHW_RGB_img_np_matplotlib(src_img_color_with_edges.transpose(2, 0, 1))

#     res = atlas_img.shape[0]

#     res = atlas_img.shape[1]
#     uint8_img = atlas_img * 255.0
#     uint8_img = np.clip(uint8_img, 0, 255)
#     uint8_img = uint8_img.astype(np.uint8)
#     # detect edges:

#     b, g, r = cv2.split(uint8_img)

#     # edges_b = cv2.Canny(b, 50, 150)
#     # edges_g = cv2.Canny(g, 50, 150)
#     # edges_r = cv2.Canny(r, 50, 150)
#     # weights = [0.33, 0.33, 0.33]
#     # edges = np.average([edges_b, edges_g, edges_r], axis=0, weights=weights)
#     im1rx = cv2.Scharr(r, cv2.CV_64F, 1, 0)
#     im1ry = cv2.Scharr(r, cv2.CV_64F, 0, 1)
#     im1rx = cv2.convertScaleAbs(im1rx)
#     im1ry = cv2.convertScaleAbs(im1ry)
#     edges_r = cv2.addWeighted(im1rx, 0.5, im1ry, 0.5, 0)

#     im1gx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
#     im1gy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
#     im1gx = cv2.convertScaleAbs(im1gx)
#     im1gy = cv2.convertScaleAbs(im1gy)
#     edges_g = cv2.addWeighted(im1gx, 0.5, im1gy, 0.5, 0)

#     im1bx = cv2.Scharr(b, cv2.CV_64F, 1, 0)
#     im1by = cv2.Scharr(b, cv2.CV_64F, 0, 1)
#     im1bx = cv2.convertScaleAbs(im1bx)
#     im1by = cv2.convertScaleAbs(im1by)
#     edges_b = cv2.addWeighted(im1bx, 0.5, im1by, 0.5, 0)

#     weights = [0.33, 0.33, 0.33]
#     edges = np.average([edges_b, edges_g, edges_r], axis=0, weights=weights)
#     print('edges.shape', edges.shape)



#     edge_mask = edges > edge_thresh  # res,res
#     edge_mask = edge_mask * ~background_edges_mask

#     # get border area by dilating edges
#     dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
#     dilate_iterations = 1
#     border_mask = cv2.dilate(edge_mask.astype(np.uint8) * 255, dilate_kernel,
#                              iterations=dilate_iterations).astype(np.bool_)  # [res,res]

#     # vis = True
#     # if vis:
#     #     edges_binary = edges.copy()
#     #     edges_binary[edges > edge_thresh] = 0
#     #     edges_binary[edges <= edge_thresh] = 255  # res,res
#     #     src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)  # [H,W,3]
#     #     src_img_color_with_edges = src_img_color.copy()  # [H,W,3]
#     #     src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  # [H,W,3]
#     #     cat = cat_images(src_img_color_with_edges.transpose(2, 0, 1), edge_mask[np.newaxis, ...])
#     #     cat = cat_images(cat,np.repeat((border_mask.astype(np.float)/1.0)[np.newaxis, ...],3,0))
#     #     display_CHW_RGB_img_np_matplotlib(cat)

#     return edge_mask,border_mask # both [res,res]



# def get_connected_components_in_atlax_view_id(per_pixel_view_id,view_num = 8):
#     '''

#         :param per_pixel_view_id:
#         :param edge_mask: [res,res]
#         :param view_num:
#         :return:
#         '''
#     res = per_pixel_view_id.shape[0]
#     foreground_mask = per_pixel_view_id > -1

#     if per_pixel_view_id.max() + 1 > view_num:
#         view_num = per_pixel_view_id.max() + 1

#     total_connected_area_num = 0
#     all_connected_area_labels = -np.ones((res, res)).astype(np.int)

#     for i in range(view_num):
#         this_view_mask = per_pixel_view_id == i
#         view_img = np.zeros((res, res)).astype(np.uint8)
#         view_img[this_view_mask] = 1
#         num_labels, labels = cv2.connectedComponents(view_img, connectivity=8)
#         num_labels -= 1 # we don't need background to have any label
#         back_ground_label = labels[~foreground_mask][0]

#         all_connected_area_labels[labels != back_ground_label] = labels[
#                                                                      labels != back_ground_label] + total_connected_area_num
#         total_connected_area_num += num_labels

#     # vis = True
#     # if vis:
#     #     colors = np.random.randint(0, 255, size=(total_connected_area_num, 3), dtype=np.uint8)
#     #     color_img = np.zeros((res, res, 3), dtype=np.uint8)
#     #
#     #     for i in range(total_connected_area_num):
#     #         color_img[all_connected_area_labels == i] = colors[i]
#     #
#     #     display_CHW_RGB_img_np_matplotlib(color_img.transpose(2, 0, 1) / 255.0)
#     return total_connected_area_num,all_connected_area_labels


# def get_notable_area_in_atlax_view_id(per_pixel_view_id,view_num = 8):
#     '''

#     :param per_pixel_view_id:
#     :param edge_mask: [res,res]
#     :param view_num:
#     :return:
#     '''
#     res = per_pixel_view_id.shape[0]
#     foreground_mask = per_pixel_view_id > -1



#     if per_pixel_view_id.max()-1 > view_num:
#         view_num = per_pixel_view_id.max()-1



#     total_connected_area_num = 0
#     all_connected_area_labels = np.zeros((res,res)).astype(np.int)




#     for i in range(view_num):
#         this_view_mask = per_pixel_view_id == i
#         view_img = np.zeros((res,res)).astype(np.uint8)
#         view_img[this_view_mask] = 1
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(view_img, connectivity=8)

#         back_ground_label = np.argmax(stats[:,4])
#         all_connected_area_labels[labels!=back_ground_label] = labels[labels!=back_ground_label]+total_connected_area_num
#         total_connected_area_num += num_labels



#     colors = np.random.randint(0, 255, size=(total_connected_area_num, 3), dtype=np.uint8)
#     color_img = np.zeros((res, res, 3), dtype=np.uint8)
#     uv_view_id_notable_area_mask = np.zeros((res, res), dtype=np.bool_)



#     pixel_num_thresh = 25000#25000 #  999999999
#     for i in range(total_connected_area_num):
#         pixel_num = (all_connected_area_labels==i).astype(np.int32).sum()

#         if pixel_num< pixel_num_thresh:
#             color_img[all_connected_area_labels == i] = colors[i]
#             uv_view_id_notable_area_mask[all_connected_area_labels == i] = True


#     # display_CHW_RGB_img_np_matplotlib(color_img.transpose(2, 0, 1) / 255.0)
#     return uv_view_id_notable_area_mask


# def get_notable_area_in_atlax_color(atlax_img,edge_thresh=100,pixel_num_thresh=25000,area_expand_thresh=3):
#     '''

#     :param atlax_img: [res,res,3]
#     :param edge_thresh:
#     :return:
#     '''
#     res = atlax_img.shape[1]
#     uint8_img = atlax_img * 255.0
#     uint8_img = np.clip(uint8_img, 0, 255)
#     uint8_img = uint8_img.astype(np.uint8)
#     # detect edges:
#     print('uint8_img.shape',uint8_img.shape)
#     b, g, r = cv2.split(uint8_img)


#     # edges_b = cv2.Canny(b, 50, 150)
#     # edges_g = cv2.Canny(g, 50, 150)
#     # edges_r = cv2.Canny(r, 50, 150)
#     # weights = [0.33, 0.33, 0.33]
#     # edges = np.average([edges_b, edges_g, edges_r], axis=0, weights=weights)
#     im1rx = cv2.Scharr(r, cv2.CV_64F, 1, 0)
#     im1ry = cv2.Scharr(r, cv2.CV_64F, 0, 1)
#     im1rx = cv2.convertScaleAbs(im1rx)
#     im1ry = cv2.convertScaleAbs(im1ry)
#     edges_r = cv2.addWeighted(im1rx, 0.5, im1ry, 0.5, 0)

#     im1gx = cv2.Scharr(g, cv2.CV_64F, 1, 0)
#     im1gy = cv2.Scharr(g, cv2.CV_64F, 0, 1)
#     im1gx = cv2.convertScaleAbs(im1gx)
#     im1gy = cv2.convertScaleAbs(im1gy)
#     edges_g = cv2.addWeighted(im1gx, 0.5, im1gy, 0.5, 0)

#     im1bx = cv2.Scharr(b, cv2.CV_64F, 1, 0)
#     im1by = cv2.Scharr(b, cv2.CV_64F, 0, 1)
#     im1bx = cv2.convertScaleAbs(im1bx)
#     im1by = cv2.convertScaleAbs(im1by)
#     edges_b = cv2.addWeighted(im1bx, 0.5, im1by, 0.5, 0)

#     weights = [0.33, 0.33, 0.33]
#     edges = np.average([edges_b, edges_g, edges_r], axis=0, weights=weights)
#     print('edges.shape',edges.shape)

#     src_img_color = uint8_img #cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)

#     edges_binary = edges.copy().astype(np.uint8)
#     edges_binary[edges > edge_thresh] = 0
#     edges_binary[edges <= edge_thresh] = 255
#     num_labels, labels = cv2.connectedComponents(edges_binary, connectivity=8)

#     colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
#     color_img = np.zeros((res, res, 3), dtype=np.uint8)

#     dilate_kernel = np.ones((3, 3), np.uint8)

#     abnormal_mask = np.zeros((res, res)).astype(np.bool_)

#     # print('-----------------------------------------------')
#     for i in range(num_labels):
#         abnormal = False
#         pixel_num = (labels == i).astype(np.int32).sum()
#         # print('pixel_num',pixel_num)
#         if pixel_num < pixel_num_thresh:  #  should be smaller than threshold
#             label_area_mask = labels == i
#             dilated_label_area_mask = cv2.dilate(label_area_mask.astype(np.uint8), dilate_kernel,
#                                                  iterations=area_expand_thresh).astype(np.bool_)

#             final_label_area_mask = dilated_label_area_mask



#             abnormal = True
#             abnormal_mask[final_label_area_mask] = True

#             color_img[final_label_area_mask] = colors[i]


#         if not abnormal:
#             color_img[labels == i] = src_img_color[labels == i]

#     src_img_color_with_edges = src_img_color.copy()
#     src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  #
#     cat = cat_images(color_img.transpose(2,0,1)/255.0, src_img_color_with_edges.transpose(2,0,1)/255.0)
#     display_CHW_RGB_img_np_matplotlib(cat)


# def get_inconsistent_area_in_atlax_color(single_view_atlas_imgs,single_view_atlas_masks,diff_thresh = 0.25):
#     # single_view_atlas_masks # (view_num,res, res, 3)
#     # single_view_atlas_imgs # (view_num,res, res, 3)
#     per_pixel_cross_view_min = np.min(single_view_atlas_imgs,
#                                       where=single_view_atlas_masks,
#                                       initial=100, axis=0)  # (res, res, 3)
#     per_pixel_cross_view_max = np.max(single_view_atlas_imgs,
#                                       where=single_view_atlas_masks,
#                                       initial=-100, axis=0)  # (res, res, 3)
#     per_pixel_cross_view_diff = per_pixel_cross_view_max - per_pixel_cross_view_min  # (res, res, 3)
#     per_pixel_cross_view_diff[per_pixel_cross_view_diff == -200] = 0  # (res, res, 3)

#     # print('per_pixel_cross_view_diff', per_pixel_cross_view_diff.min(), per_pixel_cross_view_diff.max(),
#     #       per_pixel_cross_view_diff) # 0, about 0.5, 0.6


#     diff_area = per_pixel_cross_view_diff.sum(-1) > diff_thresh  # # res,res





#     # vis = False
#     # if vis:
#     # diff_img = per_pixel_cross_view_diff[::-1, :, :].transpose(2, 0, 1)
#     #     diff_thresh_img = np.repeat(diff_area[np.newaxis, ...], 3, axis=0).astype(np.float)
#     #     print('diff_img.shape', diff_img.shape, diff_img.min(), diff_img.max())
#     #     print('diff_thresh_img.shape', diff_thresh_img.shape, diff_thresh_img.min(), diff_thresh_img.max())
#     #
#     #
#     #     cat = cat_images(diff_img.copy(), diff_thresh_img.copy())
#     #     display_CHW_RGB_img_np_matplotlib(cat)

#     return diff_area

# def fix_notable_view_id(per_pixel_view_id,single_view_atlas_masks,
#                         num_labels,labels,
#                         component_ratio_thresh =0.3, component_pixel_num_thresh = 100,
#                         view_num = 8,):
#     '''

#     :param per_pixel_view_id: [res,res]
#     :param view_num:
#     :return:
#     '''
#     res = per_pixel_view_id.shape[0]
#     refined_per_pixel_view_id = per_pixel_view_id.copy()
#     foreground_mask = per_pixel_view_id >-1

#     background_label = labels[~foreground_mask][0]

#     if per_pixel_view_id.max() + 1 > view_num:
#         view_num = per_pixel_view_id.max() + 1

#     uint8_img = per_pixel_view_id * math.floor((255.0 / (view_num - 1)))
#     uint8_img = np.clip(uint8_img, 0, 255).astype(np.uint8)


#     ### Get charts
#     num_charts, per_pixel_chart_ids = cv2.connectedComponents(foreground_mask.astype(np.uint8)*255, connectivity=8)
#     background_chart_id = per_pixel_chart_ids[~foreground_mask][0]
#     # display_CHW_RGB_img_np_matplotlib(per_pixel_chart_ids[np.newaxis,...]/num_charts)

#     for chart_id in range(num_charts):
#         if chart_id == background_chart_id:
#             continue
#         # Get how many pixels are there in this chart totally
#         chart_pixel_total_num = (per_pixel_chart_ids == chart_id).sum()

#         # Get the most frequent seen view id in this chart
#         chart_atlax_view_id = per_pixel_view_id.copy()
#         chart_atlax_view_id[per_pixel_chart_ids != chart_id] = view_num + 1
#         chart_per_view_id_num = np.bincount(chart_atlax_view_id.reshape(-1))
#         # chart_most_view_id = chart_per_view_id_num[:-1].argmax()
#         chart_per_view_id_num = chart_per_view_id_num[:view_num]
#         chart_most_view_id = chart_per_view_id_num.argmax()
#         # print('chart_most_view_id',chart_most_view_id)
#         # display_CHW_RGB_img_np_matplotlib(chart_atlax_view_id[np.newaxis,...]/num_charts)

#         chart_component_labels = labels.copy()
#         chart_component_labels[per_pixel_chart_ids != chart_id] = -1


#         for component_id in np.unique(chart_component_labels):
#             if component_id == -1:
#                 continue


#             component_mask = chart_component_labels == component_id
#             component_pixel_num = component_mask.sum()

#             if component_pixel_num/chart_pixel_total_num < component_ratio_thresh or \
#                 component_pixel_num<component_pixel_num_thresh:
#                 # this part is surrounded by neighbors
#                 # see if visible,if so, use this
#                 visible_in_most_view_id_mask = np.logical_and(single_view_atlas_masks[chart_most_view_id][:,:,0],component_mask)
#                 refined_per_pixel_view_id[visible_in_most_view_id_mask] = chart_most_view_id





#         # chart_atlax_view_id = per_pixel_view_id.copy()
#         # chart_atlax_view_id[per_pixel_chart_ids != chart_id] = -1
#         # in_chart_view_ids = np.unique(chart_atlax_view_id)
#         #
#         #
#         #
#         # chart_atlax_uint8_img = uint8_img.copy()
#         #
#         #
#         # chart_atlax_uint8_img[per_pixel_chart_ids!=chart_id] = 0
#         #
#         #
#         #
#         #
#         #
#         # ### Segment
#         #
#         # for i in in_chart_view_ids:
#         #     if i <0:
#         #         continue









#         ### Get edges
#         # edges = detect_edges_in_gray_by_scharr(uint8_img)
#         # edge_thresh = math.floor((255.0 / (view_num - 1))) - 1
#         # edge_mask = edges > edge_thresh  # res,res
#         # edge_mask = edge_mask * ~background_edges_mask



#         # vis = True
#         # if vis:
#         #     edges_binary = edges.copy()
#         #     edges_binary[edges > edge_thresh] = 0
#         #     edges_binary[edges <= edge_thresh] = 255 # res,res
#         #     src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)  # [H,W,3]
#         #     src_img_color_with_edges = src_img_color.copy() # [H,W,3]
#         #     src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  # [H,W,3]
#         #     cat = cat_images(src_img_color_with_edges.transpose(2, 0, 1),edge_mask[np.newaxis,...])
#         #     display_CHW_RGB_img_np_matplotlib(cat)
#         #
#         # # get border area by dilating edges
#         # dilate_kernel = np.ones((11, 11), np.uint8)
#         # dilate_iterations = 1
#         # border_mask = cv2.dilate(edge_mask.astype(np.uint8) * 255, dilate_kernel,
#         #                          iterations=dilate_iterations).astype(np.bool_)  # [res,res]
#         #
#         # ### Segment
#         #

#     return refined_per_pixel_view_id

