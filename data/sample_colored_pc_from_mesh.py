# taken from the following link with a few modifications:
# https://github.com/NVIDIAGameWorks/kaolin/blob/6fdb91394f6ef0c991da7c845918fb26832c5991/examples/recipes/preprocess/fast_mesh_sampling.py

# from count import get_shapenet2_mesh_data
import sys
sys.path.append("..")
sys.path.append(".")
import os
import numpy as np
from plyfile import PlyData, PlyElement
from utils.vtk_basic import vis_actors_vtk, get_colorful_pc_actor_vtk
import traceback
from utils.logger_util import get_logger
import platform
sys_platform = platform.platform().lower()
import datetime
import pytz # time zone
import torch
import nvdiffrast
import nvdiffrast.torch as dr
device = torch.device('cuda')
try:
    glctx = nvdiffrast.torch.RasterizeGLContext(False, device=device) #
except:
    glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)


# cameras are used to delete invisible points
from utils.camera_utils import create_cameras
cameras,base_dirs,eye_positions,up_dirs = create_cameras(num_views=20, distance =1.6,res =256,
                                                         device = device)



import kaolin as kal
import torch


shapenet_root_path = '/home/datasets/ShapeNetCore.v2'
shapenet_clean_root_path = '/home/datasets/shapenet_core_v2/meshes'
shapenet_sampled_pc_root_path =  '/home/datasets/ShapeNetCore.v2.pc_test'



######################################################################
#                           core code
######################################################################

# used for sampling points
def preprocessing_transform(inputs,face_visibility=None):
    """This the transform used in shapenet dataset __getitem__.
    Three tasks are done:
    1) Get the areas of each faces, so it can be used to sample points
    2) Get a proper list of RGB diffuse map
    3) Get the material associated to each face

    The inputs should contain:
    input.data is a kaolin mesh
    input.attributes['name'] should be a string
    """
    mesh = inputs.data

    vertices = mesh.vertices.unsqueeze(0)
    faces = mesh.faces
    # calculate normal # added by Qiao
    face_normals = kal.ops.mesh.face_normals(face_vertices=mesh.vertices[mesh.faces].unsqueeze(0), unit=True) # [1,num_faces,3)


    # Some materials don't contain an RGB texture map, so we are considering the single value
    # to be a single pixel texture map (1, 3, 1, 1)

    # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
    # uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0) % 1, (0, 0, 0, 1)) * 2. - 1.
    # uvs[:, :, 1] = -uvs[:, :, 1]
    uvs = torch.nn.functional.pad(mesh.uvs.unsqueeze(0),
                                  (0, 0, 0, 1))  # mod 1; *2-1, oposite ::1 should be applied after sampling points

    face_uvs_idx = mesh.face_uvs_idx
    materials_order = mesh.materials_order

    DEBUG = False
    if DEBUG:
        for i,m in enumerate(mesh.materials):
            print(m)
    materials = [
        m['map_Kd'].permute(2, 0, 1).unsqueeze(0).float() / 255. if 'map_Kd' in m else
        m['Kd'].reshape(1, 3, 1, 1)
        for m in mesh.materials
    ]

    nb_faces = faces.shape[0]
    num_consecutive_materials = \
        torch.cat([
            materials_order[1:, 1],
            torch.LongTensor([nb_faces])
        ], dim=0) - materials_order[:, 1]

    face_material_idx = kal.ops.batch.tile_to_packed(
        materials_order[:, 0],
        num_consecutive_materials
    ).squeeze(-1)
    mask = face_uvs_idx == -1
    face_uvs_idx[mask] = 0
    face_uvs = kal.ops.mesh.index_vertices_by_faces(
        uvs, face_uvs_idx
    )
    face_uvs[:, mask] = 0.


    if face_visibility is not None:
        faces = faces[face_visibility]
        face_uvs = face_uvs[:,face_visibility,:,:] # [1, face_num,3,2]
        face_material_idx = face_material_idx[face_visibility]

    # print('face_normals',face_normals,face_normals.shape) # [1,num_faces,3)

    outputs = {
        'vertices': vertices, # [1,vertex_num,3]
        'faces': faces, # [face_num,3]
        'face_areas': kal.ops.mesh.face_areas(vertices, faces), # 1,face_num
        'face_uvs': face_uvs, # [1,face_num,3,2]
        'materials': materials,
        'face_material_idx': face_material_idx, # [face_num]
        'name': inputs.attributes['name'],
        'face_normals': face_normals # [1,num_faces,3)
    }

    return outputs


# used for sampling points
class SamplePointsTransform(object):
    """

    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        # print('uv max',inputs['face_uvs'].max().item())
        coords, face_idx, feature_uvs = kal.ops.mesh.sample_points(
            inputs['vertices'],
            inputs['faces'],
            num_samples=self.num_samples,
            areas=inputs['face_areas'],
            face_features=inputs['face_uvs']
        )
        coords = coords.squeeze(0)
        face_idx = face_idx.squeeze(0)


        # we apply a modulo 1 on the UVs because ShapeNet follows GL_REPEAT behavior (see: https://open.gl/textures)
        feature_uvs = feature_uvs.squeeze(0)

        # Interpolate the RGB values from the texture map
        point_materials_idx = inputs['face_material_idx'][face_idx]
        all_point_colors = torch.zeros((self.num_samples, 3))

        uvs = feature_uvs
        uvs = (uvs % 1) * 2 - 1
        uvs[:, 1] = -uvs[:, 1]
        for i, material in enumerate(inputs['materials']):
            mask = point_materials_idx == i
            point_color = torch.nn.functional.grid_sample(
                material,
                uvs[mask].reshape(1, 1, -1, 2),
                mode='bilinear',
                align_corners=False,
                padding_mode='border')
            all_point_colors[mask] = point_color[0, :, 0, :].permute(1, 0)

        normals = inputs['face_normals'].squeeze(0)[face_idx]

        outputs = {
            'coords': coords,
            'face_idx': face_idx,
            'material_idx': point_materials_idx,
            'uvs': feature_uvs,
            'colors': all_point_colors,
            'name': inputs['name'],
            'normals': inputs['face_normals'].squeeze(0)[face_idx]
        }
        return outputs


def save_one_mesh_npy(inputs, save_root=None):
    '''
    Saved rgb colors are uint8 ranging from 0 - 255.
    :param inputs:
    :param save_root:
    :return:
    '''
    if save_root is None:
        save_root = shapenet_sampled_pc_root_path
    # save result
    cls_id, mesh_id = inputs['name'].split('/')
    # print(cls_id,mesh_id)
    save_path = os.path.join(save_root, cls_id, mesh_id)
    os.makedirs(save_path, exist_ok=True)

    if torch.is_tensor( inputs['coords']):
        coords = inputs['coords'].numpy().astype('f4')
        colors = (inputs['colors'].numpy() * 255).astype('uint8')
        normals = inputs['normals'].numpy().astype('f4')
        uvs = inputs['uvs'].numpy().astype('f4')
        material_idx = inputs['material_idx'].numpy().astype('uint8')
        face_idx = inputs['face_idx'].numpy().astype('uint8')
    else:
        coords = inputs['coords'].astype('f4')
        colors = (inputs['colors'] * 255).astype('uint8')
        normals = inputs['normals'].astype('f4')
        uvs = inputs['uvs'].astype('f4')
        material_idx = inputs['material_idx'].astype('uint8')
        face_idx = inputs['face_idx'].astype('uint8')
    # print('colors',colors)
    np.save(os.path.join(save_path, 'coords.npy'), coords)
    np.save(os.path.join(save_path, 'colors.npy'), colors)
    np.save(os.path.join(save_path, 'normals.npy'), normals)
    np.save(os.path.join(save_path, 'uvs.npy'), uvs)
    np.save(os.path.join(save_path, 'material_idx.npy'), material_idx)
    np.save(os.path.join(save_path, 'face_idx.npy'), face_idx)


# sample a given mesh; version before 2023.07.13
def sample_one_mesh(mesh_data, point_num=30000, save_root=None,skip_exist = True):
    '''
    :param mesh_data: : a KaolinDatasetItem. Can be obtained by kal.io.shapenet.ShapeNetV2()[i].
            mesh_data.data:  a kaolin.io.obj.ObjMesh, and
            mes_data.attributes: some attributes
    :param point_num: how many points to sample from this mesh
    :return:
    '''
    if skip_exist:
        if save_root is None:
            save_root = shapenet_sampled_pc_root_path
        cls_id, mesh_id = mesh_data.attributes['name'].split('/')
        save_path = os.path.join(save_root, cls_id, mesh_id)
        if os.path.exists(os.path.join(save_path, 'coords.npy')) and \
            os.path.exists(os.path.join(save_path, 'colors.npy')) and\
            os.path.exists(os.path.join(save_path, 'uvs.npy')) and\
            os.path.exists(os.path.join(save_path, 'material_idx.npy')) and\
            os.path.exists(os.path.join(save_path, 'face_idx.npy')):
            print('skip exist', save_path)
            return

    samplePointsTransform = SamplePointsTransform(point_num)
    temp = preprocessing_transform(mesh_data)
    # print(outputs.keys())
    # print(outputs['name'])
    # print(outputs['vertices'].shape) # [1, V_NUM, 3]
    # print(outputs['faces'].shape) # [F_NUM, 3]
    # print(outputs['face_areas'].shape) # [1,F_NUM]
    # print(outputs['face_uvs'].shape) # [1,F_NUM,3,2]
    # print(outputs['face_material_idx'].shape) # [F_NUM,]
    # print(outputs['materials'])

    outputs = samplePointsTransform(temp)
    # print(outputs['coords'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['face_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['colors'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['material_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['uvs'].shape)  # [sampled_V_NUM,  2]
    # print(outputs['name'])

    save_one_mesh_npy(outputs, save_root=save_root)

    return outputs

# wrapper function, sample a given mesh; version before 2023.07.13
def sample_one_mesh_(mesh_data):
    sample_one_mesh(mesh_data, point_num=30000, save_root=None)

# sample a given mesh without saving, version_2023.07.13
def sample_pc(mesh_data,point_num = 30000,face_visibility = None):
    samplePointsTransform = SamplePointsTransform(point_num)
    temp = preprocessing_transform(mesh_data,face_visibility)
    outputs = samplePointsTransform(temp)
    coords = outputs['coords'].numpy()
    colors = outputs['colors'].numpy()
    normals = outputs['normals'].numpy()
    material_idx = outputs['material_idx'].numpy()
    face_idx = outputs['face_idx'].numpy()
    uvs = outputs['uvs'].numpy()
    # print(outputs['coords'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['face_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['colors'].shape)  # [sampled_V_NUM,  3]
    # print(outputs['material_idx'].shape)  # [sampled_V_NUM]
    # print(outputs['uvs'].shape)  # [sampled_V_NUM,  2]
    # print(outputs['name'])

    return coords, colors, normals,material_idx, face_idx, uvs

# 2023.07.13
def sample_one_mesh_w_o_invisible_points(mesh_data,point_per_shape,cameras,device,save_root):
    # # load mesh
    # mesh_data = get_shapenet2_mesh_data(shapenet_clean_root_path, cls_id, name)

    mesh = mesh_data.data
    faces = mesh.faces
    vertices = mesh.vertices

    # # normalize mesh to -0.5 0.5
    mesh_v_max = vertices.max(0)[0]  # [0] if tensor
    mesh_v_min = vertices.min(0)[0]
    mesh_v_center = (mesh_v_max + mesh_v_min) / 2
    mesh_v_size = mesh_v_max - mesh_v_min
    mesh_v_size_max = mesh_v_size.max()
    vertices -= mesh_v_center
    vertices /= mesh_v_size_max

    # sample points
    p_coords, p_colors, p_normals, p_raw_material_ids, p_face_ids, p_uvs = sample_pc(mesh_data, point_num=point_per_shape*5)

    # delete invisible points by depth
    transformed_points = torch.zeros((len(cameras), p_coords.shape[0], 3), device=device)
    pos = torch.zeros((len(cameras), vertices.shape[0], 4), device=device)
    for i_cam, cam in enumerate(cameras):
        transformed_points[i_cam] = cam.transform(torch.tensor(p_coords).to(device))
        transformed_vertices = cam.transform(vertices.to(device))
        pos[i_cam] = torch.nn.functional.pad(  # Create a fake W (See nvdiffrast documentation)
            transformed_vertices, (0, 1), mode='constant', value=1.
        ).contiguous().squeeze(0).float()

    rast = nvdiffrast.torch.rasterize(glctx, pos, faces.int().to(device), (cam.height, cam.width),
                                      grad_db=False)

    normalized_depth = rast[0][..., 2]  # [num_cameras,res,res]
    res = cam.height

    # print('transformed_points.shape',transformed_points.shape) # [num_cameras,piont_num,3]
    # print('normalized_depth.shape',normalized_depth.shape)# [num_cameras,res,res]
    num_cameras = len(cameras)
    point_num = p_coords.shape[0]

    point_pixels = transformed_points[..., :2] * res / 2 + res / 2  # [num_cameras,piont_num,2]
    point_pixels = point_pixels.long()
    point_pixels = torch.cat((point_pixels[:, :, 1].unsqueeze(-1), point_pixels[:, :, 0].unsqueeze(-1)),
                             dim=-1)  # switch x and y
    point_pixels = point_pixels.clip(0, res - 1)

    point_depth = transformed_points[:, :, 2]  # # [num_cameras,point_num]
    reference_depth = normalized_depth[
        torch.arange(num_cameras).view(-1, 1),  # cam_num, 1
        point_pixels[:, :, 0].long(),  # cam_num, point_num
        point_pixels[:, :, 1].long()  # cam_num, point_num
    ]
    point_visibility = point_depth <= reference_depth
    # print('point_visibility.shape',point_visibility.shape) # [num_cameras,point_num]


    point_visibility = point_visibility.float().max(0)[0].bool()  # [point_num]
    # print('point_visibility.sum()', point_visibility.sum())


    p_coords = p_coords[point_visibility.cpu().numpy()]
    p_colors = p_colors[point_visibility.cpu().numpy()]
    p_normals = p_normals[point_visibility.cpu().numpy()]
    p_raw_material_ids = p_raw_material_ids[point_visibility.cpu().numpy()]
    p_face_ids = p_face_ids[point_visibility.cpu().numpy()]
    p_uvs = p_uvs[point_visibility.cpu().numpy()]

    # subsample points to target number
    assert (p_coords.shape[0] >= point_per_shape)
    p_ids = np.arange(p_coords.shape[0])
    np.random.shuffle(p_ids)
    selected_p_ids = p_ids[:point_per_shape]
    p_coords = p_coords[selected_p_ids]
    p_colors = p_colors[selected_p_ids]
    p_normals = p_normals[selected_p_ids]
    p_raw_material_ids = p_raw_material_ids[selected_p_ids]
    p_face_ids = p_face_ids[selected_p_ids]
    p_uvs = p_uvs[selected_p_ids]

    # Normalize colors
    # p_colors = (p_colors / 255.).astype(np.float32)
    outputs = {
        'coords': p_coords,
        'face_idx': p_face_ids,
        'material_idx': p_raw_material_ids,
        'uvs': p_uvs,
        'colors': p_colors,
        'normals': p_normals,
        'name': mesh_data.attributes['name'] #.split('/')
    }

    save_one_mesh_npy(outputs, save_root=save_root)
    return p_coords, p_colors, p_raw_material_ids, p_face_ids, p_uvs

def sample_one_mesh_w_o_invisible_points_(mesh_data):
    sample_one_mesh_w_o_invisible_points(mesh_data,point_per_shape=30000,cameras=cameras,device=device,save_root=None)



######################################################################
#                           for each dataset
######################################################################
def get_shapenet2_mesh_data(shapenet_clean_root_path, cls_id, name):
    file_name = os.path.join(shapenet_clean_root_path, cls_id, name, 'models', 'model_normalized.obj')
    mesh = kal.io.obj.import_mesh(file_name, with_materials=True)
    model_data = kal.io.dataset.KaolinDatasetItem(mesh, {'name': f'{cls_id}/{name}'})
    return model_data

def get_other_mesh_data(root_path,name,cls_id = 'google_scanned_objects'):
    file_name = os.path.join(root_path,'meshes',cls_id, name, 'models', f'{name}.obj')
    if not os.path.exists(file_name):
        file_name = os.path.join(root_path, 'meshes', cls_id, name, 'models', f'model_normalized.obj')
    if not os.path.exists(file_name):
        file_name = os.path.join(root_path, 'meshes', cls_id, name, 'meshes', f'model.obj')
    if not os.path.exists(file_name):
        file_name = os.path.join(root_path, 'meshes', cls_id, name, 'Scan', f'Scan.obj') # omni
    mesh = kal.io.obj.import_mesh(file_name, with_materials=True)
    model_data = kal.io.dataset.KaolinDatasetItem(mesh, {'name': f'{cls_id}/{name}'})
    return model_data

def sample_shapenet_core_v2_mesh_batch(batch_num = 1):
    from multiprocessing import Pool
    import time
    now = time.strftime("%Y_%m_%d %H.%M.%S\n", time.localtime())
    logger = get_logger(f'data/{now}_sample_pc_by_kaolin.log')
    ds = kal.io.shapenet.ShapeNetV2(root=shapenet_clean_root_path,
                                    categories={'02958343','03790512'}, # 03001627,02958343,03790512
                                    train=False,
                                    split=0,
                                    with_materials=True
                                    )

    model_num = len(ds)
  
    for i in range(0,model_num):

        logger.info(
            f"{i}/{model_num}:{ds[i].attributes['name']}")
        try:
            # sample_one_mesh_(ds[i])
            sample_one_mesh_w_o_invisible_points_(ds[i])
        except KeyboardInterrupt:
            logger.error((traceback.format_exc()))
            sys.exit(0)
        except:
            logger.error(traceback.format_exc())


def sample_google_scanned_objects_batch(root_path = 'datasets/google_scanned_objects',cls_id = 'google_scanned_objects'):
    now = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M.%S')
    logger = get_logger(os.path.join(root_path,f'{now}_sample_pc.log'))

    names = os.listdir(os.path.join(root_path,'meshes',cls_id))
    model_num = len(names)
    # for i in range(10): # for testing only
    for i in range(0, model_num):  # for real batch sampling
        mesh_data = get_other_mesh_data(root_path,names[i],cls_id=cls_id)
        logger.info(
            f"{i}/{model_num}:{mesh_data.attributes['name']}")
        try:
            # sample_one_mesh_(ds[i])
            sample_one_mesh_w_o_invisible_points(mesh_data,point_per_shape=30000,cameras=cameras,device=device,
                                                 save_root=os.path.join(root_path,'pc_kaolin'))
        except KeyboardInterrupt:
            logger.error((traceback.format_exc()))
            sys.exit(0)
        except:
            logger.error(traceback.format_exc())

def sample_omniobject3d_batch(root_path = 'datasets/omniobject3d',cls_id = 'omniobject3d'):
    now = datetime.datetime.now(pytz.timezone('Etc/GMT-8')).strftime('%Y.%m.%d.%H.%M.%S')
    logger = get_logger(os.path.join(root_path,f'{now}_sample_pc.log'))

    names = os.listdir(os.path.join(root_path,'meshes',cls_id))
    model_num = len(names)
    # for i in range(10): # for testing only
    for i in range(0, model_num):  # for real batch sampling
        mesh_data = get_other_mesh_data(root_path,names[i],cls_id=cls_id)
        logger.info(
            f"{i}/{model_num}:{mesh_data.attributes['name']}")
        try:
            # sample_one_mesh_(ds[i])
            sample_one_mesh_w_o_invisible_points(mesh_data,point_per_shape=30000,cameras=cameras,device=device,
                                                 save_root=os.path.join(root_path,'pc_kaolin'))
        except KeyboardInterrupt:
            logger.error((traceback.format_exc()))
            sys.exit(0)
        except:
            logger.error(traceback.format_exc())

######################################################################
#                           test code
######################################################################
# get mesh_data; for testing only



# test by dataset
def test_sample_by_datset():
    ds = kal.io.shapenet.ShapeNetV2(root=shapenet_clean_root_path,
                                    categories={'03001627'},
                                    train=False,
                                    split=0,
                                    with_materials=True
                                    )
    for i in range(len(ds)):
        model_data = ds[i]
        print(i, model_data.attributes['name'])
        # output = sample_one_mesh(model_data, 100000)
        output = sample_one_mesh_w_o_invisible_points_(model_data)
        vis_now = True
        if vis_now:
            vis_actors_vtk([
                get_colorful_pc_actor_vtk(output['coords'], point_colors=output['colors'] * 255, opacity=1)
            ])


def recover_color_from_uv(shapenet_clean_root_path, shapenet_sampled_pc_root_path,cls_id, name):
    # load data

    # pc_file_name = os.path.join(sampled_pc_root_path, cls_id, f'{name}.ply')
    # coords, gt_colors, material_ids, face_ids, uvs = read_ply(pc_file_name)
    pc_file_dir = os.path.join(shapenet_sampled_pc_root_path, cls_id, name)
    coords, gt_colors, material_ids, face_ids, uvs = load_pc_npy(pc_file_dir)
    # print(uvs.shape) # [30000,2]
    # print(gt_colors)
    # vis_actors_vtk([
    #     get_colorful_pc_actor_vtk(coords, point_colors=gt_colors , opacity=1)
    # ])

    mesh_file_name = os.path.join(shapenet_clean_root_path, cls_id, name, 'models', 'model_normalized.obj')
    if not os.path.exists(mesh_file_name):
        mesh_file_name = os.path.join(shapenet_clean_root_path, cls_id, name, 'models', f'{name}.obj')
    mesh = kal.io.obj.import_mesh(mesh_file_name, with_materials=True)
    materials = [
        m['map_Kd'].permute(2, 0, 1).unsqueeze(0).float() / 255. if 'map_Kd' in m else
        m['Kd'].reshape(1, 3, 1, 1)
        for m in mesh.materials
    ]

    num_samples = coords.shape[0]
    print('num samples',num_samples)
    # get colors from uvs
    colors = torch.zeros((num_samples, 3))
    uvs = (uvs % 1) * 2 - 1
    uvs[:, 1] = -uvs[:, 1]
    for i, material in enumerate(materials):
        mask = material_ids == i
        point_color = torch.nn.functional.grid_sample(
            material,
            torch.tensor(uvs[mask].reshape(1, 1, -1, 2)),
            mode='bilinear',
            align_corners=False,
            padding_mode='border')
        colors[mask] = point_color[0, :, 0, :].permute(1, 0)
    diff = gt_colors / 255. - colors.numpy()
    print('diff', diff, diff.max(), diff.min())
    # print('gt',gt_colors)
    # print('recover',colors)
    # # vis
    vis_actors_vtk([
        get_colorful_pc_actor_vtk(coords - np.array([1,0,0]), point_colors=colors *255, opacity=1),
        get_colorful_pc_actor_vtk(coords + np.array([1,0,0]), point_colors=gt_colors  , opacity=1)
    ])


def test_recover_color_from_uv():
    files = os.listdir(os.path.join(shapenet_sampled_pc_root_path, '03001627'))
    for i, file in enumerate(files):
        print(i, file)
        # # coords,colors,material_ids,face_ids,uvs = read_ply(os.path.join(sampled_pc_root_path,'03001627',file))
        # coords, colors, material_ids, face_ids, uvs = load_pc_npy(
        #     os.path.join(sampled_pc_root_path, '03001627', file))
        # vis_actors_vtk([
        #     get_colorful_pc_actor_vtk(coords, point_colors=colors , opacity=1)
        # ])

        recover_color_from_uv(shapenet_clean_root_path, shapenet_sampled_pc_root_path,cls_id='03001627', name=file.split('.ply')[0])


######################################################################
#                          util code
######################################################################



def load_pc_npy(filedir):
    coords = np.load(os.path.join(filedir, 'coords.npy'))
    colors = np.load(os.path.join(filedir, 'colors.npy'))
    uvs = np.load(os.path.join(filedir, 'uvs.npy'))
    material_idx = np.load(os.path.join(filedir, 'material_idx.npy'))
    face_idx = np.load(os.path.join(filedir, 'face_idx.npy'))
    return coords, colors, material_idx, face_idx, uvs


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    # print((pc))
    # print(type(pc))

    coords = np.array([[x, y, z] for x, y, z, r, g, b, m, f, u, v in pc])
    colors = np.array([[r, g, b] for x, y, z, r, g, b, m, f, u, v in pc])
    material_ids = np.array([m for x, y, z, r, g, b, m, f, u, v in pc])
    # face_ids = np.array([f for x,y,z,r,g,b,m,f,u,v in pc])
    uvs = np.array([[u, v] for x, y, z, r, g, b, m, f, u, v in pc])
    face_ids = None
    # uvs=None
    return coords, colors, material_ids, face_ids, uvs


def save_one_mesh_ply(inputs, save_root=None):
    if save_root is None:
        save_root = shapenet_sampled_pc_root_path

    # save result
    cls_id, mesh_id = inputs['name'].split('/')
    # print(cls_id,mesh_id)
    os.makedirs(os.path.join(save_root, cls_id), exist_ok=True)

    pc = inputs['coords'].numpy()
    colors = inputs['colors'].numpy() * 255
    uvs = inputs['uvs'].numpy()
    # print('uvs',uvs,uvs.max(),uvs.min())
    material_idx = inputs['material_idx'].numpy()
    face_idx = inputs['face_idx'].numpy()

    vertices = np.empty(pc.shape[0],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8'),
                               ('m', 'uint8'), ('f', 'uint8'),
                               ('u', 'f4'), ('v', 'f4'),
                               ])
    vertices['x'] = pc[:, 0].astype('f4')
    vertices['y'] = pc[:, 1].astype('f4')
    vertices['z'] = pc[:, 2].astype('f4')

    vertices['red'] = colors[:, 0].astype('f4')
    vertices['green'] = colors[:, 1].astype('f4')
    vertices['blue'] = colors[:, 2].astype('f4')

    vertices['m'] = material_idx.astype('int8')
    vertices['f'] = face_idx.astype('int8')

    vertices['u'] = uvs[:, 0].astype('f4')
    vertices['v'] = uvs[:, 1].astype('f4')

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    path = os.path.join(shapenet_sampled_pc_root_path, cls_id, f'{mesh_id}.ply')
    ply.write(path)




if __name__ == '__main__':


    sample_shapenet_core_v2_mesh_batch() # sample
    test_recover_color_from_uv()  # check result


    sample_google_scanned_objects_batch()
    sample_omniobject3d_batch()
