
print('start import')
import datetime
import sys

from tqdm import tqdm

from baselines.spr import recon_one_shape_SPR
sys.path.append('models/POCO')
import shutil
import torch
device = torch.device('cuda')
import kaolin as kal
import nvdiffrast
# try:
#     glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device) #
# except:
#     glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
from utils.logger_util import get_logger
from munch import Munch
import yaml
from pointdreamer.ours_utils import *
from pointdreamer.unproject import unproject,dilate_atlas,paint_invisible_areas_by_optimize,paint_invisible_areas_by_neighbors
from utils.other_utils import read_ply_xyzrgb,save_colored_pc_ply
from utils.camera_utils import create_cameras
from utils.utils_2d import save_CHW_RGBA_img,dilate_foreground
from models.POCO.generate_1 import POCO_get_geo,create_POCO_network,POCO_config
import time

import argparse
import kiui
print('finish import')
kiui.seed_everything(42)



def colorize_one_mesh(
        # name,root_path,cls_id,
        #             load_exist_dense_img_path,
                    coords,colors, # input point cloud
                    vertices,faces,f_normals, # untextured mesh

                    xatlas_dict,
                    camera_info,
                    view_num,res,cam_res,refine_res,
                     #
                    device,
                    save_img_path,

                    point_validation_by_o3d,refine_point_validation_by_remove_abnormal_depth,hidden_point_removal_radius,


                    texture_gen_method,point_size,edge_point_size,

                    crop_img, crop_padding,mask_ratio_thresh,
                    optimize_from,
                    edge_dilate_kernels,
                    complete_unseen_by,
                  

                    inpainter,glctx,
                    logger,

                    xatlas_texture_res,


                    **kwargs):
    
    base_dirs = camera_info['base_dirs']
    cams = camera_info['cams']
    cam_RTs = camera_info['cam_RTs']
    cam_K = camera_info['cam_K']
    eye_positions = camera_info['eye_positions']
    up_dirs = camera_info['up_dirs']

    uvs = xatlas_dict['uvs']
    mesh_tex_idx = xatlas_dict['mesh_tex_idx']
    gb_pos = xatlas_dict['gb_pos']
    mask = xatlas_dict['mask']
    per_atlas_pixel_face_id = xatlas_dict['per_atlas_pixel_face_id']

    with torch.no_grad():
       

        projected_points = coords
   


        ''' Prepare data for projecting (rendering) 3D to 2D'''

        start_project = time.time()
        hard_masks, face_idxs, mesh_normalized_depths, vertice_uvs, uv_centers, uv_scales,padding,point_uvs,point_depths = \
            get_rendered_hard_mask_and_face_idx_batch(cams, vertices, faces, projected_points,glctx=glctx,
                                                      rescale=crop_img,padding=crop_padding)




        '''get muti-view images'''
       
        # Hidden Point Removal
        if cam_res != res:
            hard_masks = transforms.Resize((res,res))(hard_masks.unsqueeze(1).float()).squeeze(1).bool() # N,cam_res,cam_res


        point_validation1,_ = get_point_validation_by_depth(cam_res,point_uvs,point_depths,mesh_normalized_depths)
        point_validation2 = get_point_validation_by_o3d(projected_points,eye_positions,hidden_point_removal_radius)
        point_validation = torch.logical_or(point_validation1,point_validation2)
        
        

        if refine_point_validation_by_remove_abnormal_depth: # False by default
            point_validation = refine_point_validation(cam_RTs,cam_K, refine_res,
                            hard_masks, point_validation, point_uvs, projected_points,save_img_path)


        # get sparse img
        point_pixels = point_uvs * res   # [num_cameras,piont_num,2]
        point_pixels = point_pixels.long()
        point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                                    dim=-1)  # switch x and y
        point_pixels = point_pixels.clip(0, res - 1)
  
        sparse_imgs,hard_mask0s,hard_mask2s,inpaint_scale_factors = get_sparse_images\
            (point_pixels,colors,point_validation,hard_masks,save_img_path,view_num,res,point_size,edge_point_size,
                mask_ratio_thresh)

      
        try:
            logger.info(f'project: {time.time()-start_project} s')
        except:
            pass
        # get dense img
        start_inpainting = time.time()
        exist_inpainted_multiview_imgs=True # TODO
        if exist_inpainted_multiview_imgs: # load from exist
            inpainted_images = torch.zeros((len(cams), 3, res, res)).to(device)
            for i in range(view_num):

                inpainted_img_path = os.path.join(save_img_path, f'{i}_inpainted.png')
                if os.path.exists(inpainted_img_path):
                    inpainted_images[i] = load_CHW_RGB_img(inpainted_img_path).to(device)
                else:
                    exist_inpainted_multiview_imgs=False

        if not exist_inpainted_multiview_imgs: # inpaint now
            inpainted_images = get_inpainted_images(sparse_imgs,hard_mask0s,hard_mask2s,save_img_path, inpainter,view_num,
                                                    method=texture_gen_method)
            

            try:
                logger.info(f'inpainting: {time.time() - start_inpainting} s')
            except:
                pass
        # # dilate the foreground of the images to avoid artifacts at the edge
        # inpainted_images_rgba = torch.cat([inpainted_images,hard_mask0s[:,:1,:,:]],dim=1) # N3HW, N3HW -> N4WH
        # inpainted_images_rgba = dilate_foreground(inpainted_images_rgba)
        # inpainted_images = inpainted_images[:,:3]
        # for i in range(view_num):
        #     # hard_mask0s: [view_num,3,res,res]
        #     inpainted_rgba = torch.cat([inpainted_images[i],hard_mask0s[i][0].unsqueeze(0)])
            
        #     save_CHW_RGBA_img( inpainted_rgba.cpu().numpy(), os.path.join(save_img_path, f'{i}.png'))

    ''' Unproject inpainted 2D rendered images back to 3D'''
    start_unproject = time.time()
    complete_unseen_by_projection = (complete_unseen_by=='unproject')
    atlas_img,shrinked_per_view_per_pixel_visibility,point_view_ids,points_atlas_pixel_coord,points,atlas_painted_mask = \
    unproject(inpainted_images,vertices,f_normals,
                res,
                cams,cam_res,base_dirs,
                gb_pos,mask,per_atlas_pixel_face_id,
                uv_centers,uv_scales,padding,inpaint_scale_factors,
                mesh_normalized_depths,edge_dilate_kernels,save_img_path,complete_unseen_by_projection)
    # point_view_ids = torch.ones_like(point_view_ids).to(device)*-100.0 # use this to directly predicting colors by 3D optimization
    # 
    to_inpaint_face_id = per_atlas_pixel_face_id[0][torch.logical_not(atlas_painted_mask)].unique()
    to_inpaint_face_id = to_inpaint_face_id[to_inpaint_face_id>-1]
    ## Complete unseen areas that cannot be seen from any view
    if complete_unseen_by=='optimize':
        atlas_img = paint_invisible_areas_by_optimize(atlas_img,points,points_atlas_pixel_coord,point_view_ids,input_xyz=coords,input_rgb=colors)
        atlas_img = dilate_atlas(atlas_img,mask)
    elif complete_unseen_by =='neighbor':
        use_atlas = True
        if not use_atlas:
            subdivided_vertices,subdivided_faces,subdevided_vert_colors = paint_invisible_areas_by_neighbors(
            vertices,faces,uvs,mesh_tex_idx,to_inpaint_face_id,atlas_img, atlas_painted_mask,use_atlas=False)
            mesh = trimesh.Trimesh(
                vertices=subdivided_vertices.cpu().numpy(), 
                faces=subdivided_faces.cpu().numpy(), 
                vertex_colors=subdevided_vert_colors.cpu().numpy(),
            )
            mesh.export('subdivided.obj', 'obj')
            return
        else:
            atlas_img = paint_invisible_areas_by_neighbors(
                vertices,faces,uvs,mesh_tex_idx,to_inpaint_face_id,atlas_img, atlas_painted_mask,use_atlas=True)
   
    elif complete_unseen_by=='unproject':
        atlas_img = dilate_atlas(atlas_img,mask)    

    try:
        logger.info(f'unporject before optimize: {time.time() - start_unproject} s')
    except:
        pass
    
    ## further optimize the result 
    if optimize_from is not None:
        start_optimzie = time.time()
        if optimize_from != 'None':
            # print('1. atlas_img.shape',atlas_img.shape) # [res,res,3]
            eye_positions = torch.tensor(eye_positions).float().to(device)
            # up_dirs = torch.tensor(up_dirs).float().to(device)
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
            # for i in range(view_num):
            #     save_CHW_RGB_img(final_render_result[i].detach().cpu().numpy(),
            #                         os.path.join(root_path, 'meshes', cls_id, name, 'models',
            #                                     f'final_render_result_{i}.png'))

            # print('2. atlas_img.shape', atlas_img.shape)
        try:
            logger.info(f' optimize: {time.time() - start_optimzie} s')
        except:
            pass

    try:
        logger.info(f'unproject: {time.time() - start_unproject} s')
    except:
        pass

    return vertices,uvs,faces,mesh_tex_idx,atlas_img,mask
    

        

    
    




def save_textured_mesh(vertices,uvs,faces,mesh_tex_idx,atlas_img,mask,output_root_path):
    '''
    atlas_image: [res,res,3]
    mask: [1,res,res,1], bool
    '''

    # save mesh
    savemeshtes2(
        vertices.data.cpu().numpy(), # pointnp_px3
        uvs.data.cpu().numpy(), # tcoords_px2
        faces.data.cpu().numpy(), # facenp_fx3
        mesh_tex_idx.data.cpu().numpy(), # facetex_fx3

        os.path.join(output_root_path, 'models', 'model_normalized.obj') # fname
    )

    # save texture image

    # tex_map = atlas_img_visibility
    lo, hi = (0, 1)
    tex_map = atlas_img
    img = np.asarray(tex_map.data.cpu().numpy(), dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    # dilate_mask = mask[0] # from [1,res,res,1] to [1,res,res]
    # # dilate_mask = dilate_mask.detach().cpu().numpy()
    # img *= dilate_mask.detach().cpu().numpy()  # added by PointDreamer author, necessary to enable later dialate
    # img = img.clip(0, 255)
    # dilate_mask = np.sum(img.astype(float), axis=-1,
    #                 keepdims=True)  # mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
    # dilate_mask = (dilate_mask <= 3.0).astype(float)  # mask = (mask <= 3.0).astype(np.float)
    # kernel = np.ones((3, 3), 'uint8')
    # dilate_img = cv2.dilate(img, kernel, iterations=1)  # without this, some faces will have edges with wrong colors
    # img = img * (1 - dilate_mask) + dilate_img * dilate_mask
    img = img.clip(0, 255).astype(np.uint8)

    print('img.shape',img.shape)
    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
        os.path.join(output_root_path, 'models', 'model_normalized.png'))
    
    # Also save atlas without background
    cat_mask = (mask[0].long()*255).detach().cpu().numpy().astype(np.uint8) # from [1,res,res,1] to [res,res,1]
    rgba_atlas_img = np.concatenate([img,cat_mask],axis=-1)
    PIL.Image.fromarray(np.ascontiguousarray(rgba_atlas_img[::-1, :, :]), 'RGBA').save(
        os.path.join(output_root_path, 'others', 'atlas_wo_background.png'))
    


def prepare(cfg_file):
    
    torch.backends.cudnn.benchmark = True
    # load cfg
    cfg_txt = open(cfg_file, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    # create logger
    logger=get_logger(os.path.join(cfg.output_path,f'{datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")}_log.log'))

    # load inpainter
    inpainter = None
    if cfg.texture_gen_method == 'DDNM_inpaint':
        logger.info('Loading inpainter...')
        from models.DDNM.ddnm_inpainting import Inpainter

        inpainter = Inpainter(device)
        logger.info('inpainter loaded')
    
    # load cameras
    cams,base_dirs,eye_positions,up_dirs = create_cameras(num_views=cfg.view_num,
                                            distribution = cfg.camera_distribution,
                                                distance =1.6,res =cfg.cam_res,device= device)
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
    camera_info = {
        'cams':cams,
        'base_dirs':base_dirs,
        'eye_positions':eye_positions,
        'up_dirs':up_dirs,
        'cam_RTs':cam_RTs,
        'cam_K':cam_K
    }
    logger.info('Loading POCO...')
    POCO_net = create_POCO_network(POCO_config)
    
    return cfg,inpainter, POCO_net, camera_info, logger


def recon_one_textured_mesh(cfg,inpainter,POCO_net,camera_info,pc_file,name):
    # makd dirs
    output_path = cfg.output_path
    untextured_mesh_path = os.path.join(output_path,name,'geo')
    xatlas_save_file = os.path.join(output_path,name, 'geo',f'xatlas_{cfg.xatlas_texture_res}.pth')
    os.makedirs(untextured_mesh_path,exist_ok=True)
    os.makedirs(os.path.join(output_path,name,'models'),exist_ok=True)
    os.makedirs(os.path.join(output_path,name,'geo'),exist_ok=True)
    os.makedirs(os.path.join(output_path,name,'others'),exist_ok=True)

    # load input colored pc and normalize
    xyz,rgb = read_ply_xyzrgb(pc_file)
    if len(xyz)>30000:
        print(len(xyz))
        print(f"Point number > 30000! ({len(xyz)} points)({pc_file}) \n Please try uniformly subsampling the input point cloud first")
        raise NotImplementedError
    xyz = torch.tensor(xyz).to(device)
    rgb = torch.tensor(rgb).float().to(device) /255.0
    vertices_min = xyz.min(0)[0]
    vertices_max = xyz.max(0)[0]
    xyz -= (vertices_max + vertices_min) / 2.
    xyz /= (vertices_max - vertices_min).max()



    # Get Geometry
    load_exist_geo = True
    all_start = time.time()
    start = time.time()
    possible_geo_path = pc_file.replace('.ply','_untextured_mesh.obj')
    # print(os.path.exists(possible_geo_path),possible_geo_path)
    if load_exist_geo:
        if os.path.exists(possible_geo_path):
            mesh = kal.io.obj.import_mesh(possible_geo_path)
            vertices = mesh.vertices.to(device)
            faces = mesh.faces.to(device).long()
            vertices -= (vertices_max + vertices_min) / 2.
            vertices /= (vertices_max - vertices_min).max()
        else:
            logger.info('Get Geometry by POCO...')
            untextured_mesh_file = os.path.join(untextured_mesh_path,name+'_untextured','models','model_normalized.obj')
            if os.path.exists(untextured_mesh_file):
                mesh = kal.io.obj.import_mesh(untextured_mesh_file)
                vertices = mesh.vertices.to(device)
                faces = mesh.faces.to(device).long()
            else:
                load_exist_geo = False
    if not load_exist_geo:
        if cfg.geo_from == 'SPR':
            vertices,faces,_ = recon_one_shape_SPR(xyz.detach().cpu().numpy(),
                                                   rgb.detach().cpu().numpy(),
                                                   simplify_face_num=10000)
            vertices = torch.tensor(vertices).float().to(device).contiguous()
            faces = torch.tensor(faces).to(device).long().contiguous()
        else:  # POCO
            vertices,faces = POCO_get_geo(POCO_config,xyz,POCO_net,savedir_mesh_root = untextured_mesh_path, object_name=name+'_untextured')

    f_normals = kal.ops.mesh.face_normals(face_vertices=vertices[faces].unsqueeze(0), unit=True)[0] # [ F, 3]
    logger.info(f'Get Geometry time: {time.time()-start} s by {cfg.geo_from}')
          
    # un unwrapping
    start = time.time()
    logger.info('UV Unwrapping by xatlas...')
    load_exist_xatlas = True
    if load_exist_xatlas:
        if os.path.exists(xatlas_save_file):  # load
            try:
                xatlas_dict = torch.load(xatlas_save_file)
                logger.info('Existing xatlas data loaded')
            except:
                load_exist_xatlas = False
        else:
            load_exist_xatlas = False
    
    if not load_exist_xatlas:  # calculate and save
        logger.info('Conducting UV Unwrapping...')
        uvs, mesh_tex_idx, gb_pos, mask,per_atlas_pixel_face_id = xatlas_uvmap_w_face_id(
            glctx, vertices, faces, resolution=cfg.xatlas_texture_res)
        import kiui

        xatlas_dict = {'uvs': uvs, 'mesh_tex_idx': mesh_tex_idx, 'gb_pos': gb_pos, 
                       'mask': mask,'per_atlas_pixel_face_id':per_atlas_pixel_face_id}
        # os.makedirs(os.path.dirname(xatlas_save_path), exist_ok=True)
        torch.save(xatlas_dict, xatlas_save_file)
    logger.info(f'xatlas time: {time.time()-start} s')
          
    # generate texture by PointDreamer
    
    logger.info('Generate texture by PointDreamer...')
    start = time.time()
    vertices,uvs,faces,mesh_tex_idx,atlas_img,mask = colorize_one_mesh(xyz,rgb,vertices,faces,f_normals,
                                                                       
                                                                       xatlas_dict,
                                                                      
                                                                       camera_info,
                                                                       inpainter=inpainter,
                                                                       save_img_path = os.path.join(output_path,name,'others'),
                                                                       device=device,logger=None,glctx=glctx,**cfg)
    logger.info(f'generate texture time: {time.time()-start} s')
    
    save_textured_mesh(vertices,uvs,faces,mesh_tex_idx,atlas_img,mask,os.path.join(output_path,name))
    logger.info(f'total time: {time.time()-all_start} s')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("PointDreamer")
    parser.add_argument("--config", type=str, help="path to config file",default='configs/default.yaml')
    parser.add_argument("--pc_file", type=str, help="path to input point cloud file",default ='dataset/demo_data/clock.ply')
    args = parser.parse_args()
    cfg_file = args.config
    cfg,inpainter,POCO_net,camera_info,logger = prepare(cfg_file)
    
    if args.pc_file.endswith('.ply'):
        pc_files = [args.pc_file]
    else:
        pc_root_path = args.pc_file
        pc_files = os.listdir(pc_root_path)
        pc_files = [os.path.join(pc_root_path,i) for i in pc_files if i.endswith('.ply')]
        

    for pc_file in tqdm(pc_files):
        name = os.path.basename(pc_file).split('.')[0] + '_' + os.path.basename(cfg_file).split('.')[0]

        os.makedirs(os.path.join(cfg.output_path,name),exist_ok=True)
        shutil.copy(cfg_file,os.path.join(cfg.output_path,name,'config.yaml'))
        logger.info(f'Start Recon {pc_file}...')
        recon_one_textured_mesh(cfg,inpainter,POCO_net,camera_info, pc_file,name)

    '''
    Example usage:
    python demo.py --config configs/default.yaml --pc_file dataset/demo_data/clock.ply
    python demo.py --config configs/default.yaml --pc_file dataset/demo_data/PaulFrankLunchBox.ply
    python demo.py --config configs/default.yaml --pc_file dataset/demo_data/rolling_lion.ply

    python demo.py --config configs/default.yaml --pc_file dataset/NBF_demo_data/2ce6_chair.ply
    python demo.py --config configs/wo_NBF.yaml --pc_file dataset/NBF_demo_data/2ce6_chair.ply

    '''


    