import sys
sys.path.append('..')
sys.path.append('.')
from utils.camera_utils import render_textured_mesh,render_textured_meshes_shapenet2,render_per_vertex_color_meshes_shapenet2
import torch
import nvdiffrast
import os
import argparse


'''
The file organization should be like:
- rootpath
    - meshes
        - cls_id
            - shape_name
                - models
                    -model_normalized.obj
    - rendered_imgs
        - cls_id
            - shape_name
                - albedo_0XX.png # XX is from 01 to 20
                
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser("render meshes")
    parser.add_argument("--rootpath", type=str,default='', help="path to the root path of meshes to be rendered")
    parser.add_argument("--by", type=str,default='kaolin', help="kaolin or kaolin_per_vertex")
    args = parser.parse_args()
    rootpath = args.rootpath
    by = args.by



    if by == 'kaolin':
        device = torch.device('cuda')
        try:
            glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device)  #
        except:
            glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        cls_ids = os.listdir(os.path.join(rootpath,'meshes'))

        # save_root_path = render_path = mesh_root_path.replace('meshes','rendered_imgs')
        save_root_path = render_path = rootpath
        render_textured_meshes_shapenet2(root_path=rootpath, device=device, glctx=glctx)

    elif by =='kaolin_per_vertex':
        device = torch.device('cuda')
        try:
            glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device)  #
        except:
            glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)
        cls_ids = os.listdir(os.path.join(rootpath, 'meshes'))

        # save_root_path = render_path = mesh_root_path.replace('meshes','rendered_imgs')
        save_root_path = render_path = rootpath
        render_per_vertex_color_meshes_shapenet2(mesh_root_path=rootpath, device=device, glctx=glctx)

