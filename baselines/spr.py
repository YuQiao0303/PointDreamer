import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import os
import numpy as np
import time
import sys

from tqdm import tqdm
from utils.logger_util import get_logger

import traceback

import datetime
import pytz # time zone
import shutil

from utils.other_utils import read_ply_xyzrgb

import pymeshlab




def recon_one_shape_SPR(coords_np_Vx3, colors_np_Vx3,
                        gt_normals_np_Vx3=None,save_path = None,depth = 8 ,simplify_face_num=None):
    '''
    By default we use depth=12 for higher-quality reconstruction.
    If it's too slow to run on your device, try modify it from 12 to 8. 
    '''
    # https://pymeshlab.readthedocs.io/en/2022.2/classes/mesh.html
    # https://pymeshlab.readthedocs.io/en/2022.2/tutorials/import_mesh_from_arrays.html

    coords = coords_np_Vx3
    colors = colors_np_Vx3
    normals = gt_normals_np_Vx3
    use_GT_normals = gt_normals_np_Vx3 is not None
    save_obj = save_path is not None
    # create a new MeshSet object
    ms = pymeshlab.MeshSet()

    # load a point cloud by pymeshlab
    # ms.load_new_mesh('pointcloud.ply')
    colors_4 = np.concatenate((colors, np.ones((colors.shape[0], 1))), axis=1)  # N,3 -> N,4

    if use_GT_normals:
        m = pymeshlab.Mesh(vertex_matrix = coords, 
                           v_normals_matrix = normals,v_color_matrix = colors_4)
        ms.add_mesh(m)  # add the mesh to the MeshSet
    else:
        print('estimate normals')
        m = pymeshlab.Mesh(vertex_matrix=coords,v_color_matrix=colors_4)
        ms.add_mesh(m)  # add the mesh to the MeshSet
        ms.apply_filter('compute_normal_for_point_clouds')


    print('start poisson reconstruction, this may take some time...')
    # apply the screened poisson filter
    ms.apply_filter('generate_surface_reconstruction_screened_poisson', depth=depth)

    # simplify mesh
    if simplify_face_num is not None:
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=simplify_face_num, preservetopology=True)
    
    if save_obj:
        ms.save_current_mesh(save_path) # save obj

    vertices = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix()
    vertice_colors = ms.current_mesh().vertex_color_matrix() # V,4

    return vertices,faces,vertice_colors[:,:3]

    
    
def main():
    parser = argparse.ArgumentParser("SPR_baseline")
    parser.add_argument("--pc_file", type=str, help="path to input point cloud file",default ='dataset/demo_data/clock.ply')
    args = parser.parse_args()
    output_path = 'output_baseline/SPR'
  
    
    if args.pc_file.endswith('.ply'):
        pc_files = [args.pc_file]
    else:
        pc_root_path = args.pc_file
        pc_files = os.listdir(pc_root_path)
        pc_files = [os.path.join(pc_root_path,i) for i in pc_files if i.endswith('.ply')]
        

    for pc_file in tqdm(pc_files):
        name = os.path.basename(pc_file).split('.')[0] 
        os.makedirs(os.path.join(output_path,name),exist_ok=True)
   
        coords_np_Vx3,colors_np_Vx3_uint8 = read_ply_xyzrgb(pc_file)
        colors_np_Vx3 = colors_np_Vx3_uint8.astype(np.float64) / 255.0
        recon_one_shape_SPR(coords_np_Vx3=coords_np_Vx3,colors_np_Vx3=colors_np_Vx3,
                            gt_normals_np_Vx3=None,
                            save_path = os.path.join(output_path,name,f'{name}.obj'))

    '''
    Example usage:
    python baselines/spr.py  --pc_file dataset/demo_data/clock.ply
    python baselines/spr.py  --pc_file dataset/demo_data/
    '''


if __name__ == '__main__':
    main()