from torch_geometric.data import Dataset
from lightconvpoint.datasets.data import Data
import os
import numpy as np
import torch
import logging

class FamousTest(Dataset):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, variant_directory="famous_original", dataset_size=None, **kwargs):
            
        super().__init__(root, transform, None)

        logging.info(f"Dataset  - Famous Test - Test only - {dataset_size}")


        self.root = os.path.join(self.root, variant_directory)

        self.filenames = []
        split_file = os.path.join(self.root, "testset.txt")

        with open(split_file) as f:
            content = f.readlines()
            content = [line.split("\n")[0] for line in content]
            content = [os.path.join(self.root, "04_pts", line) for line in content]
            self.filenames += content
        self.filenames.sort()

        if dataset_size is not None:
            self.filenames = self.filenames[:dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")

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
        raise NotImplementedError
        data_shape = np.load(os.path.join(filename, "pointcloud.npz"))
        data_space = np.load(os.path.join(filename, "points.npz"))
        return data_shape, data_space

    def get(self, idx):
        """Get item."""
        filename = self.filenames[idx]

        pts_shp = np.load(filename+".xyz.npy")
        pts_shp = torch.tensor(pts_shp, dtype=torch.float)
        pts_space = torch.ones((1,3), dtype=torch.float)
        occupancies = torch.ones((1,), dtype=torch.long)

        data = Data(x = torch.ones_like(pts_shp),
                    shape_id=idx, 
                    pos=pts_shp,
                    pos_non_manifold=pts_space, occupancies=occupancies, #
                    )

        return data



class FamousTestNoiseFree(FamousTest):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, variant_directory="famous_noisefree", dataset_size=None, **kwargs):
            
        super().__init__(root, 
            split=split, 
            transform=transform,
            filter_name=filter_name, 
            num_non_manifold_points=num_non_manifold_points, 
            variant_directory=variant_directory, 
            dataset_size=dataset_size, **kwargs)

class FamousTestExtraNoisy(FamousTest):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, variant_directory="famous_extra_noisy", dataset_size=None, **kwargs):
            
        super().__init__(root, 
            split=split, 
            transform=transform,
            filter_name=filter_name, 
            num_non_manifold_points=num_non_manifold_points, 
            variant_directory=variant_directory, 
            dataset_size=dataset_size, **kwargs)

class FamousTestSparse(FamousTest):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, variant_directory="famous_sparse", dataset_size=None, **kwargs):
            
        super().__init__(root, 
            split=split, 
            transform=transform,
            filter_name=filter_name, 
            num_non_manifold_points=num_non_manifold_points, 
            variant_directory=variant_directory, 
            dataset_size=dataset_size, **kwargs)

class FamousTestDense(FamousTest):

    def __init__(self, root, split="training", transform=None, filter_name=None, num_non_manifold_points=2048, variant_directory="famous_dense", dataset_size=None, **kwargs):
            
        super().__init__(root, 
            split=split, 
            transform=transform,
            filter_name=filter_name, 
            num_non_manifold_points=num_non_manifold_points, 
            variant_directory=variant_directory, 
            dataset_size=dataset_size, **kwargs)