from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import glob
import logging

class ShapeNet(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, dataset_size=None, **kwargs):
            
        super().__init__(root, transform, None) # root: data/ShapeNet

        logging.info(f"Dataset  - ShapeNet- {dataset_size}")

        self.split = split
        self.filter_name = filter_name
        self.filelists = []
        self.num_non_manifold_points = num_non_manifold_points
     
        self.data_root_name = 'pc_kaolin'  
        print('root',root)
        print('os.path.basename(root)',os.path.basename(root))

        if os.path.basename(root) == 'ShapeNet':
            if split == 'training':
                split_file = os.path.join(self.root,'splits','train.txt')
            elif split == 'validation':
                split_file =  os.path.join(self.root,'splits','val.txt') # val
            elif split == 'test':
                split_file = os.path.join(self.root,'splits','test.txt') # test #
                np.random.seed(3407)
            with open(split_file, "r", encoding="utf-8") as file_obj:
                model_names = file_obj.read().splitlines()  # list without '\n'
            # model_names = [x for x in model_names if not x.startswith('02958343')] # debug
            # model_names = [x for x in model_names if not x.startswith('03001627')] # debug
            # model_names = [x for x in model_names if  x.startswith('03001627')] # debug
            # 03001627,  02958343

        else: # omni, google scanned objects
            cls_id = os.path.basename(root)
            model_names = os.listdir(os.path.join(root,self.data_root_name,cls_id))

          
        model_names = [cls_id + '/' +model_name for model_name in model_names]

        self.filenames = model_names
     

        logging.info(f"Dataset - len {len(self.filenames)}")


        self.metadata = {
        "google_scanned_objects": {
            "id": "google_scanned_objects",
            "name": "google_scanned_objects"
        },
        "omniobject3d": {
            "id": "omniobject3d",
            "name": "omniobject3d"
        },


        "04256520": {
            "id": "04256520",
            "name": "sofa"
        },
        "02691156": {
            "id": "02691156",
            "name": "airplane"
        },
        "03636649": {
            "id": "03636649",
            "name": "lamp"
        },
        "04401088": {
            "id": "04401088",
            "name": "phone"
        },
        "04530566": {
            "id": "04530566",
            "name": "vessel"
        },
        "03691459": {
            "id": "03691459",
            "name": "speaker"
        },
        "03001627": {
            "id": "03001627",
            "name": "chair"
        },
        "02933112": {
            "id": "02933112",
            "name": "cabinet"
        },
        "04379243": {
            "id": "04379243",
            "name": "table"
        },
        "03211117": {
            "id": "03211117",
            "name": "display"
        },
        "02958343": {
            "id": "02958343",
            "name": "car"
        },
        "02828884": {
            "id": "02828884",
            "name": "bench"
        },
        "04090263": {
            "id": "04090263",
            "name": "rifle"
        }
    }


    def get_category(self, f_id):
        return self.filenames[f_id].split("/")[-2]

    def get_object_name(self, f_id):
        return self.filenames[f_id].split("/")[-1]

    def get_class_name(self, f_id):
        return self.metadata[self.get_category(f_id)]["name"]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)


    def get_data_for_evaluation(self, idx):
        filename = self.filenames[idx]

        cls_id, shape_id = filename.split('/')

        pc = np.load(os.path.join(self.root, self.data_root_name, cls_id, shape_id, 'coords.npy'))

        normals = np.load(os.path.join(self.root, self.data_root_name, cls_id, shape_id, 'normals.npy'))

        occ_points = occupancies = None
        # points = np.load(os.path.join(self.root, 'point', cls_id, f"{shape_id}.npz"))
        #
        # occ_points = torch.tensor(points["points"], dtype=torch.float)
        # occupancies = torch.tensor(np.unpackbits(points['occupancies']), dtype=torch.long)
        return pc,normals,occ_points,occupancies

    def get(self, idx):
        """Get item."""
        filename = self.filenames[idx]
        cls_id,shape_id = filename.split('/')
        # manifold_data =np.load(os.path.join(self.root,'pointcloud',cls_id,f"{shape_id}.npz"))
        # manifold_data =np.load(os.path.join(filename, "pointcloud.npz"))
        # points_shape = manifold_data["points"]
        # normals_shape = manifold_data["normals"]
        # print(os.path.join(self.root, self.data_root_name, cls_id,shape_id, 'coords.npy'))
        points_shape = np.load(os.path.join(self.root, self.data_root_name, cls_id,shape_id, 'coords.npy'))
        if 'noisy' in self.data_root_name:
            pass
        else:  # add noise manually
            noise = np.random.normal(loc=0.0, scale=0.005, size=points_shape.shape)  # shape [N, 3]
            # print('noise',noise)
            points_shape = points_shape + noise

        try:
            normals_shape = np.load(os.path.join(self.root,self.data_root_name, cls_id,shape_id, 'normals.npy'))
        except: # for when we don't need normals
            normals_shape = points_shape
        colors_shape = np.load(os.path.join(self.root, self.data_root_name, cls_id,shape_id, 'colors.npy'))
        colors_shape = colors_shape/255.0


        pts_shp = torch.tensor(points_shape, dtype=torch.float)
        nls_shp = torch.tensor(normals_shape, dtype=torch.float)
        colors_shp = torch.tensor(colors_shape, dtype=torch.float)


        try:
            points = np.load(os.path.join(self.root,'point',cls_id,f"{shape_id}.npz"))
            # points = np.load(os.path.join(filename, "points.npz"))
            points_space = torch.tensor(points["points"], dtype=torch.float)
            occupancies = torch.tensor(np.unpackbits(points['occupancies']), dtype=torch.long)

        except: # for when we don't need occupancies
            points_space = pts_shp
            occupancies = pts_shp[:,0]
        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    normal=nls_shp,
                    color=colors_shp,
                    pos_non_manifold=points_space, occupancies=occupancies, #
                    )
        # print('pts_shp',pts_shp)

        return data