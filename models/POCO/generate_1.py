# import sys
# sys.path.append('...')
# sys.path.append('..')
# sys.path.append('.')
import os
import torch
import numpy as np
import logging
import time
import open3d as o3d
from scipy.spatial import KDTree
import torch_geometric.transforms as T
import torch.nn.functional as F

# POCO imports
import networks
from lightconvpoint.utils import transforms as lcp_T
from lightconvpoint.utils.misc import dict_to_device


from generate import export_mesh_and_refine_vertices_region_growing_v2

POCO_config = {
            # 'experiment_name': None, 'dataset_name': 'ShapeNet', 'dataset_root': 'data/google_scanned_objects', 
        #    'save_dir': '/home/me/POCO/results/POCO_google_scanned_objects', 
        #    'train_split': 'training', 'val_split': 'validation', 'test_split': 'test', 'filter_name': None, 
        #    'non_manifold_points': 2048, 'normals': False,  'random_noise': 'None', 
           'manifold_points': 30000, 
             
            #  'training_random_scale': None, 'training_random_rotation_x': None, 
            #  'training_random_rotation_y': None, 'training_random_rotation_z': None, 'training_batch_size': 16,
            #    'training_iter_nbr': 100000, 'training_lr_start': 0.001, 'resume': False,
               'network_backbone': 'FKAConv', 'network_latent_size': 32, 'network_decoder': 'InterpAttentionKHeadsNet', 
               'network_decoder_k': 64, 'network_n_labels': 2, 'device': 'cuda', #'threads': 8, 
               'log_mode': 'interactive', 'logging': 'INFO', #'val_num_mesh': None, 'val_interval': 1,
                #  'config': '/home/me/POCO/results/POCO_google_scanned_objects/config.yaml', 
                #  'config_default': 'configs/config_default.yaml', 'iter_nbr': 600000, 
                 'gen_refine_iter': 10,  'num_mesh': None, 
                

                # # scene
                # 'gen_autoscale': True, 'gen_autoscale_target': 0.01, 'gen_resolution_metric': 0.01


                # object
                'gen_subsample_manifold': 3000, 
                'gen_subsample_manifold_iter': 10,  # modify to balance performance and speed,default 10
                'gen_resolution_global': 128, # modify to balance performance and speed, default 128
                'target_number_of_triangles': 10000 # object, default, this one is added by PointDreamer author, to simplify the generated mesh
               
                 }


def create_POCO_network(config):
     
    N_LABELS = config["network_n_labels"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}
        
    net = networks.Network(3, latent_size, N_LABELS, backbone, decoder)


    checkpoint = torch.load('models/POCO/checkpoint.pth') # /home/me/POCO/results/POCO_ShapeNet_Noise/checkpoint.pth
    net.load_state_dict(checkpoint["state_dict"])
    net.to(config["device"])
    net.eval()
    return net

def POCO_get_geo(config,pc,net,savedir_mesh_root,object_name,is_noisy=False,):
    if not is_noisy: # since the pretrained weights of the POCO are trained on noisy data, we add some noise to the input point cloud
        pc = pc+ torch.randn_like(pc) * 0.005
    pc = pc.permute(1,0).unsqueeze(0) # add batch # [1,3,N]
    data={'pos':pc,'x':torch.ones_like(pc).to(pc.device)}

    device = torch.device(config["device"])


    # operate the permutations
    test_transform=[]
    test_transform = test_transform + [
                                            lcp_T.Permutation("pos", [1,0]),
                                            lcp_T.Permutation("pos_non_manifold", [1,0]),
                                            lcp_T.Permutation("normal", [1,0]),
                                            lcp_T.Permutation("x", [1,0]),
                                            lcp_T.ToDict(),]
    test_transform = T.Compose(test_transform)



    with torch.no_grad():
    
        # os.makedirs(savedir_points, exist_ok=True)
        savedir_mesh = os.path.join(savedir_mesh_root)
        os.makedirs(savedir_mesh, exist_ok=True)
        os.makedirs(os.path.join(savedir_mesh, object_name, 'models'), exist_ok=True)

        data = dict_to_device(data, device)


        start = time.time()
        # auto scale (for big scenes)
        if "gen_autoscale" in config and config["gen_autoscale"]:
            logging.info("Autoscale computation")
            autoscale_target = config["gen_autoscale_target"] # 0.01 # estimated on shapenet 3000
            pos = data["pos"][0].cpu().transpose(0,1).numpy()
            tree = KDTree(pos)
            mean_dist = tree.query(pos, 2)[0].max(axis=1).mean()
            scale = autoscale_target / mean_dist
            logging.info(f"Autoscale {scale}")
        else:
            scale = 1

        # scale the points
        data["pos"] = data["pos"] * scale


        # if too musch points and no subsample iteratively compute the latent vectors
        if data["pos"].shape[2] > 100000 and ("gen_subsample_manifold" not in config or config["gen_subsample_manifold"] is None):
            
            # create the KDTree
            pos = data["pos"][0].cpu().transpose(0,1).numpy()
            tree = KDTree(pos)

            # create the latent storage
            latent = torch.zeros((pos.shape[0], config["network_latent_size"]), dtype=torch.float)
            counts = torch.zeros((pos.shape[0],), dtype=torch.float)
            
            n_views = 3
            logging.info(f"Latent computation - {n_views} views")
            for current_value in range(0,n_views):
                while counts.min() < current_value+1:
                    valid_ids = np.argwhere(counts.cpu().numpy()==current_value)
                    # print(valid_ids.shape)
                    pt_id = torch.randint(0, valid_ids.shape[0], (1,)).item()
                    pt = pos[valid_ids[pt_id]]
                    k = 100000
                    distances, neighbors = tree.query(pt, k=k)

                    neighbors = neighbors[0]

                    data_partial = {
                        "pos": data["pos"][0].transpose(1,0)[neighbors].transpose(1,0).unsqueeze(0),
                        "x": data["x"][0].transpose(1,0)[neighbors].transpose(1,0).unsqueeze(0)
                    }

                    partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]

                    latent[neighbors] += partial_latent[0].cpu().numpy().transpose(1,0)
                    counts[neighbors] += 1

            latent = latent / counts.unsqueeze(1)
            latent = latent.transpose(1,0).unsqueeze(0).to(device)
            data["latents"] = latent
            latent = data
            logging.info("Latent done")

        elif "gen_subsample_manifold" in config and config["gen_subsample_manifold"] is not None:
                logging.info("Submanifold sampling")



                use_our_voxel_based_subsample = False
                if use_our_voxel_based_subsample:

                    pos = data["pos"][0].permute(1,0).cpu().numpy()
                    x = data["x"][0].permute(1,0).cpu().numpy()
                    N = pos.shape[0]
                    latents = []
                    for i in range(config["gen_subsample_manifold_iter"]):
                        shuffled_indices = torch.randperm(N).numpy()
                        # print('pos.shape',pos.shape)
                        # print('shuffled_indices.shape',shuffled_indices.shape)
                        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pos[shuffled_indices]))
                        pcd.colors = o3d.utility.Vector3dVector(x[shuffled_indices])
                        sub_pcd = o3d.geometry.PointCloud.farthest_point_down_sample(pcd, N //config["gen_subsample_manifold_iter"])

                        subset_pos = torch.tensor(np.asarray(sub_pcd.points)).float().to(device)
                        subset_x = torch.tensor(np.asarray(sub_pcd.colors)).float().to(device)
                        data_partial = {
                            "pos": subset_pos.permute(1,0).unsqueeze(0),
                            "x": subset_x.permute(1,0).unsqueeze(0)
                        }

                        partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]
                        latents.append(partial_latent)
                    latents = torch.stack(latents,0)
                    print('latents.shape',latents.shape)
                    latent = latents.mean(0)
                    print('latent.shape', latent.shape)
                else: # default POCO
                    # create the KDTree
                    pos = data["pos"][0].cpu().transpose(0, 1).numpy()
                    # create the latent storage
                    latent = torch.zeros((pos.shape[0], config["network_latent_size"]), dtype=torch.float)
                    counts = torch.zeros((pos.shape[0],), dtype=torch.float)


                    iteration = 0
                    for current_value in range(config["gen_subsample_manifold_iter"]):
                        while counts.min() < current_value+1:
                            # print("iter", iteration, current_value)
                            valid_ids = torch.tensor(np.argwhere(counts.cpu().numpy()==current_value)[:,0]).long()

                            if pos.shape[0] >= config["gen_subsample_manifold"]:

                                ids = torch.randperm(valid_ids.shape[0])[:config["gen_subsample_manifold"]]
                                ids = valid_ids[ids]

                                if ids.shape[0] < config["gen_subsample_manifold"]:
                                    ids = torch.cat([ids, torch.randperm(pos.shape[0])[:config["gen_subsample_manifold"] - ids.shape[0]]], dim=0)
                                assert(ids.shape[0] == config["gen_subsample_manifold"])
                            else:
                                ids = torch.arange(pos.shape[0])

                    
                            data_partial = {
                                "pos": data["pos"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0),
                                "x": data["x"][0].transpose(1,0)[ids].transpose(1,0).unsqueeze(0)
                            }
                         
                            partial_latent = net.get_latent(data_partial, with_correction=False)["latents"]
                            # print('partial_latent.shape',partial_latent.shape)
                            latent[ids] += partial_latent[0].cpu().numpy().transpose(1,0)
                            counts[ids] += 1

                            iteration += 1

                    latent = latent / counts.unsqueeze(1)
                    latent = latent.transpose(1,0).unsqueeze(0).to(device)
                data["latents"] = latent
                # print('latent.shape',latent.shape)
                latent = data
            
        else:
            # all prediction
            latent = net.get_latent(data, with_correction=False)

        
        if "gen_resolution_metric" in config and config["gen_resolution_metric"] is not None:
            step = config['gen_resolution_metric'] * scale
            resolution = None
        elif config["gen_resolution_global"] is not None:
            step = None
            resolution = config["gen_resolution_global"]
        else:
            raise ValueError("You must specify either a global resolution or a metric resolution")



        # print("POS", data["pos"].shape)
        if "target_number_of_triangles" in config:
            target_number_of_triangles=config["target_number_of_triangles"]
        else:
            target_number_of_triangles = None
        mesh = export_mesh_and_refine_vertices_region_growing_v2(
            net, latent,
            resolution=resolution,
            padding=1,
            mc_value=0,
            device=device,
            input_points=data["pos"][0].cpu().numpy().transpose(1,0),
            refine_iter=config["gen_refine_iter"],
            out_value=1,
            step=step,
            simplification_target=target_number_of_triangles
        )

        
        if mesh is not None:

            vertices = np.asarray(mesh.vertices)
            vertices = vertices / scale
            vertices = o3d.utility.Vector3dVector(vertices)
            mesh.vertices = vertices



          
            # o3d.io.write_triangle_mesh(os.path.join(savedir_mesh, object_name), mesh)
            o3d.io.write_triangle_mesh(os.path.join(savedir_mesh, object_name,
                                                    'models',"model_normalized.obj"), mesh)

        else:
            logging.warning("mesh is None")
        logging.info(f'{time.time() - start}')

        verts = torch.tensor(np.asarray(mesh.vertices)).to(device).float()
        faces = torch.tensor(np.asarray(mesh.triangles)).to(device).long()
        return verts,faces

if __name__ == '__main__':
    # from utils.other_utils import read_ply_xyz
    from plyfile import PlyData, PlyElement
    def read_ply_xyz(file):
        ply = PlyData.read(file)
        vtx = ply['vertex']

        xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
        return xyz

    device = POCO_config["device"]
    pc = read_ply_xyz('dataset/demo_data/clock/clock_pc.ply')
    pc = torch.tensor(pc).to(device)

    net = create_POCO_network(POCO_config)
    POCO_get_geo(POCO_config,pc,net,savedir_mesh_root = 'output/untextured_meshes',object_name='clock')