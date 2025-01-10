import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os

from tqdm import tqdm
import nksr
import torch
import time
import sys

import numpy as np
import pymeshlab
import traceback

import datetime
import pytz
from plyfile import PlyData, PlyElement

# from utils.other_utils import read_ply_xyzrgb, save_colored_pc_ply # requires trimesh, which may not be installed in nksr conda environment


device = torch.device("cuda:0")
reconstructor = nksr.Reconstructor(device)


def read_ply_xyzrgb(file):
    ply = PlyData.read(file)
    vtx = ply['vertex']

    xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    rgb = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=-1)

    return xyz,rgb

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


# before runnning, please prepare environment following: https://github.com/nv-tlabs/NKSR
# example usage:
# CUDA_VISIBLE_DEVICES=1 python baselines/NKSR.py  --pc_file dataset/XXX.ply
if __name__ == '__main__':
    parser = argparse.ArgumentParser("NKSR_baseline")
    parser.add_argument("--pc_file", type=str, help="path to input point cloud file",default ='dataset/demo_data/clock.ply')
    args = parser.parse_args()
    
    # logger = get_logger(os.path.join('Logs','NKSR.log'))
    now = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M.%S')
    root_path = f'output_baseline/NKSR' 
    output_path = root_path
    os.makedirs(root_path, exist_ok=True)
    use_GT_normals = False
    
    if args.pc_file.endswith('.ply'):
        pc_files = [args.pc_file]
    else:
        pc_root_path = args.pc_file
        pc_files = os.listdir(pc_root_path)
        pc_files = [os.path.join(pc_root_path,i) for i in pc_files if i.endswith('.ply')]
        

    for pc_file in tqdm(pc_files):
        name = os.path.basename(pc_file).split('.ply')[0] 
        os.makedirs(os.path.join(output_path,name),exist_ok=True)

        
        # makd dirs
        os.makedirs(os.path.join(output_path,name,'models'),exist_ok=True)

        # load input colored pc and normalize
        xyz,rgb = read_ply_xyzrgb(pc_file)
       
        xyz = torch.tensor(xyz).to(device)
        rgb = torch.tensor(rgb).float().to(device) /255.0
        vertices_min = xyz.min(0)[0]
        vertices_max = xyz.max(0)[0]
        xyz -= (vertices_max + vertices_min) / 2.
        xyz /= (vertices_max - vertices_min).max()

        # save input colored pc
        save_colored_pc_ply(xyz.detach().cpu().numpy(),rgb.detach().cpu().numpy(),
                            os.path.join(output_path,name,'input_pc.ply'))

        ####################################
        
        # coords_batch = xyz.unsqueeze(0)
        # colors_batch = rgb.unsuqeeze(0)
        # normals_batch = None
        
        # names = names_batch = [name]

 
        save_path = os.path.join(root_path,  name,
                                    'models')  # f'temp/samples/textured_mesh/{name}/{name}'
        obj_file = os.path.join(save_path, 'model_normalized.obj')
        mlt_file = os.path.join(save_path, 'model_normalized.mtl')
        pc_file = os.path.join(save_path, 'model.ply')

        os.makedirs(os.path.dirname(obj_file), exist_ok=True)

        if os.path.exists(obj_file):
            print(f'skip exist ', obj_file)
            continue

        coords = xyz.detach().cpu().numpy()
        colors = rgb.detach().cpu().numpy()


        # https://pymeshlab.readthedocs.io/en/2022.2/classes/mesh.html
        # https://pymeshlab.readthedocs.io/en/2022.2/tutorials/import_mesh_from_arrays.html
        start = time.time()
        # create a new MeshSet object
        ms = pymeshlab.MeshSet()

        # load a point cloud by pymeshlab
        # ms.load_new_mesh('pointcloud.ply')
        colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

        if use_GT_normals:
            m = pymeshlab.Mesh(vertex_matrix = coords, v_normals_matrix = normals,v_color_matrix = colors_4)
            ms.add_mesh(m)  # add the mesh to the MeshSet
        else:
            print('estimate normals')
            m = pymeshlab.Mesh(vertex_matrix=coords,v_color_matrix=colors_4)
            ms.add_mesh(m)  # add the mesh to the MeshSet
            ms.apply_filter('compute_normal_for_point_clouds')
            m = ms.current_mesh() # get the current mesh, otherwise m is still without normals
            normals = m.vertex_normal_matrix()

        input_xyz = torch.tensor(coords,dtype=torch.float32,device=device)
        input_normal = torch.tensor(normals,dtype=torch.float32,device=device)
        input_color = torch.tensor(colors,dtype=torch.float32,device=device)
        # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
        field = reconstructor.reconstruct(input_xyz, input_normal)
        # input_color is also a tensor of shape [N, 3]
        field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
        # Increase the dual mesh's resolution.
        mesh = field.extract_dual_mesh(mise_iter=2)



        # Save result as ply files
        ms = pymeshlab.MeshSet() # create a new MeshSet object
        vertex_colors = mesh.c.cpu().numpy()
        vertices = mesh.v.cpu().numpy()
        faces = mesh.f.cpu().numpy()
        colors_4 = np.concatenate((vertex_colors, np.ones((vertex_colors.shape[0], 1))), axis=1)  # N,3 -> N,4


        m = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix = faces, v_color_matrix=colors_4)
        ms.add_mesh(m)  # add the mesh to the MeshSet

        ms.save_current_mesh(obj_file)  # save obj
        # convert from y-up to z-up # somehow ply files are y-up while obj are z-up
        save_ply = True
        if save_ply:
            # save ply
            ms.apply_filter('compute_matrix_from_scaling_or_normalization', axisz=-1)
            ms.apply_filter('compute_matrix_from_rotation', rotaxis=0, angle=90)

            ms.save_current_mesh(obj_file.replace('.obj', '.ply'))
        print('time:',time.time()-start,'s')


       