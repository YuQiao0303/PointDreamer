# tested with torch2.0.0
pip install torch==2.0.0  torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.3%2Bpt20cu118-cp38-cp38-linux_x86_64.whl
pip install ninja xatlas gdown plyfile tensorboard scikit-image trimesh open3d munch pymcubes torch-geometric opencv-python

## Alternatively, if use torch2.1.0:
pip install torch==2.1.0  torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp310-cp310-linux_x86_64.whl
pip install ninja xatlas gdown plyfile tensorboard scikit-image trimesh open3d munch pymcubes torch-geometric opencv-python


