# Google Scanned Objects
## Ours
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/google_scanned_objects/2024.01.30.11.51.30_ours_kaolin_DDNM_inpaint_google_scanned_objects/ --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/google_scanned_objects

## SPR
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/google_scanned_objects/2024.01.29.21.57.29_spr_google_scanned_objects_kaolin/ --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/google_scanned_objects

## NKSR
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/google_scanned_objects/2024.01.31.16.26.28_NKSR_kaolin_google_scanned_objects/ --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/google_scanned_objects


# OmniObject3D


## Ours
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/omniobject3d/2024.01.24.19.23.43_ours_kaolin_DDNM_inpaint_omniobject3d --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/omniobject3d
## SPR
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/omniobject3d/2024.01.24.21.04.19_spr_omniobject3d_kaolin/ --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/omniobject3d
## NKSR
python eval_meshes.py --gendir /home/yuqiao/texrecon_by_2d/out_inference/omniobject3d/2024.01.25.15.20.03_NKSR_kaolin_omniobject3d/ --dataset ShapeNet --split test --gtdir /home/yuqiao/texrecon_by_2d/datasets/omniobject3d
