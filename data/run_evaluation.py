import sys
sys.path.join('..')
sys.path.join('.')
import os
import torch
from torch.utils.data import Dataset

from utils.metric_utils.fid import calculate_stats, calculate_frechet_distance, init_inception, forward_inception_batch
from utils.metric_utils.psnr_ssmi import calculate_psnr_batch,calculate_ssim_batch
from PIL import Image

import numpy as np
from torchvision.transforms import transforms
import glob
from tqdm import tqdm
import argparse

import lpips
import time

from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images #

def imread(filename,background=(0,255,0)):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    background: None or (r,g,b) [0,255]
    """
    img_all_channels = np.array(Image.open(filename).convert('RGBA'), dtype=np.uint8)  # [..., :3]
    img = img_all_channels[..., :3]  # HW3

    if background is not None:
        alpha_channel = img_all_channels[..., 3]
        img[alpha_channel == 0] = background # set transparent pixels to white
    img = img.transpose(2,0,1) # CHW
    img = img / 255.0

    return img




class RenderedImageDataset(Dataset,):
    def __init__(self, gt_root_path, pred_root_path, dataset_name='shapenet_core_v2', split='test',
                 cls_id = '03001627',view_num = 20, rendered_img_res = 512):
        self.gt_root_path = gt_root_path
        self.pred_root_path = pred_root_path
        self.view_num = view_num
        self.rendered_img_res = rendered_img_res

        names_here = os.listdir(os.path.join(pred_root_path, cls_id))
        if split is not None:
            self.split_file = os.path.join('datasets', dataset_name, 'splits', f'{cls_id}_{split}.txt')
            with open(self.split_file, "r", encoding="utf-8") as file_obj:
                model_names = file_obj.read().splitlines()  # list without '\n'
            self.names = model_names
            # assert len(names_here) == len(self.names)
        else:

            self.names = names_here


        for i in range(len(self.names)):
            if '/' not in self.names[i]:
                self.names[i] = f'{cls_id}/' + self.names[i]


    def __getitem__(self, idx):
        gt_images = np.zeros((self.view_num,3,self.rendered_img_res,self.rendered_img_res))
        pred_images = np.zeros((self.view_num,3,self.rendered_img_res,self.rendered_img_res))

        for view_id in range(self.view_num):
            # if self.split_file is None:
            if False:
                gt_path = os.path.join(self.gt_root_path,
                                         self.unique_cls_ids[self.cls_id_indeces[idx]],
                                         self.names[idx],'color_{:s}.png'.format(str(view_id+1).zfill(3)))
                pred_path = os.path.join(self.pred_root_path,
                                         self.unique_cls_ids[self.cls_id_indeces[idx]],
                                         self.names[idx],'albedo_{:s}.png'.format(str(view_id+1).zfill(3))) # albedo
            else: # names include cls_id
                gt_path = os.path.join(self.gt_root_path,
                                       self.names[idx], 'color_{:s}.png'.format(str(view_id + 1).zfill(3)))
                if not os.path.exists(gt_path):
                    gt_path = os.path.join(self.gt_root_path,
                                       self.names[idx], 'albedo_{:s}.png'.format(str(view_id + 1).zfill(3)))
                pred_path = os.path.join(self.pred_root_path,
                                         self.names[idx], 'albedo_{:s}.png'.format(str(view_id + 1).zfill(3)))
                if not os.path.exists(pred_path):
                    pred_path = os.path.join(self.pred_root_path,
                                             self.names[idx], 'color_{:s}.png'.format(str(view_id + 1).zfill(3)))
            gt_img = imread(gt_path)
            pred_img = imread(pred_path)
            if gt_img.shape[1]!= self.rendered_img_res:
                gt_img = transforms.Resize((self.rendered_img_res, self.rendered_img_res))(torch.tensor(gt_img)).numpy()
            if pred_img.shape[1]!= self.rendered_img_res:
                pred_img = transforms.Resize((self.rendered_img_res, self.rendered_img_res))(torch.tensor(pred_img)).numpy()
            gt_images[view_id] = gt_img
            pred_images[view_id] = pred_img

            # print(self.names[idx])
        return gt_images,pred_images

    def __len__(self):
        return len(self.names)





class Tester(object):
    def __init__(self,gt_root_path = None,pred_root_path=None,dataset_name=None,
                 view_num = 20, rendered_img_res = 512,batch_size = 1,device = torch.device('cuda',0),
                 metrics = ['FID', 'LPIPS'],cls_id = '03001627'):

        self.view_num = view_num
        self.rendered_img_res = rendered_img_res
        self.batch_size = batch_size
        self.device = device
        self.metrics = metrics

        self.lpips = -100
        self.fid = -100
        self.psnr = -100
        self.ssim = -100


        if cls_id.startswith('0'): # shapenet core v2:
            split = 'test'
        else: # other dataset
            split = None
        split=None # debug
        self.eval_set = RenderedImageDataset(gt_root_path=gt_root_path, pred_root_path=pred_root_path,
                                             dataset_name = dataset_name,
                                         view_num = self.view_num, rendered_img_res = self.rendered_img_res,
                                             cls_id=cls_id,split=split)
        self.data_loader = torch.utils.data.DataLoader(self.eval_set, batch_size=1, num_workers=1, \
                                                  pin_memory=True, drop_last=False, shuffle=False)



    def eval_lpips(self,gt_root_path,pred_root_path):
        self.lpips_loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)


        final_lpips_score = []

        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(tqdm(self.data_loader)):
            gt_images,pred_images = data
            gt_images = gt_images.to(self.device).float()
            pred_images = pred_images.to(self.device).float()

            gt_images = gt_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            pred_images = pred_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)

            with torch.no_grad():
                lpips_score = self.lpips_loss_fn_vgg(pred_images *2-1, gt_images *2-1) # RGB ranging from -1 to 1
            lpips_score = lpips_score.cpu().detach().numpy() # [batch*view, 1,1,1]
            final_lpips_score.append(lpips_score.mean())

        final_lpips_score = np.array(final_lpips_score).mean()

        print('-' * 100)
        print('gt_root_path:', gt_root_path)
        print('pred_root_path:', pred_root_path)
        print('view_num:', self.view_num)
        print('rendered_img_res:', self.rendered_img_res)
        print('-' * 100)
        print('LPIPS', final_lpips_score)
        print('-' * 100)
        self.lpips = final_lpips_score


    def eval_fid(self,gt_root_path,pred_root_path):
        self.inception_model = torch.nn.DataParallel(init_inception()).to(self.device).eval()
       
        emb_fake = []
        emb_real = []


        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(tqdm(self.data_loader)):
            gt_images,pred_images = data
            gt_images = gt_images.to(self.device).float()
            pred_images = pred_images.to(self.device).float()

            gt_images = gt_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            pred_images = pred_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)

            # fid
            emb_fake.append(forward_inception_batch(self.inception_model, pred_images))
            emb_real.append(forward_inception_batch(self.inception_model, gt_images))


        emb_fake = np.concatenate(emb_fake, axis=0)
        emb_real = np.concatenate(emb_real, axis=0)
        print('calculate_stats...')
        m1, s1 = calculate_stats(emb_fake)
        m2, s2 = calculate_stats(emb_real)
        # print('m1,s1',m1,s1)
        # print('m2, s2',m2, s2)
        print('calculate_frechet_distance...')
        fid = calculate_frechet_distance(m1, s1, m2, s2)

        print('-' * 100)
        print('gt_root_path:', gt_root_path)
        print('pred_root_path:', pred_root_path)
        print('view_num:', self.view_num)
        print('rendered_img_res:', self.rendered_img_res)
        print('-' * 100)
        print('FID', fid)
        print('-' * 100)

        self.fid = fid
        #

    def eval_psnr(self,gt_root_path,pred_root_path):
      

        final_score = []

        start = time.time()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(tqdm(self.data_loader)):

            gt_images,pred_images = data
            gt_images = (gt_images*255.0).numpy().round() .astype(np.uint8)
            pred_images =  (pred_images*255.0).numpy().round() .astype(np.uint8)


            gt_images = gt_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            pred_images = pred_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            gt_images = gt_images.transpose(0, 2, 3, 1)  # N,H,W,3 # doesn't seem necessary for psnr, same result
            pred_images = pred_images.transpose(0, 2, 3, 1)  # X,H,W,3
            psnr = calculate_psnr_batch(gt_images, pred_images, border=0)
            final_score.append(psnr)
        end = time.time()
        final_score = np.array(final_score).mean()

        print('-' * 100)
        print('time spent:',end-start,'s')
        print('gt_root_path:', gt_root_path)
        print('pred_root_path:', pred_root_path)
        print('view_num:', self.view_num)
        print('rendered_img_res:', self.rendered_img_res)
        print('-' * 100)
        print('PSNR', final_score)
        print('-' * 100)

        self.psnr = final_score

    def eval_ssim(self,gt_root_path,pred_root_path):


        final_score = []

        start = time.time()
        # for i, data in enumerate(tqdm(data_loader)):
        for i, data in enumerate(tqdm(self.data_loader)):
            gt_images,pred_images = data
            gt_images = (gt_images*255.0).numpy().round() .astype(np.uint8)
            pred_images =  (pred_images*255.0).numpy().round() .astype(np.uint8)


            gt_images = gt_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            pred_images = pred_images.reshape(self.batch_size*self.view_num, 3, self.rendered_img_res,self.rendered_img_res)
            gt_images = gt_images.transpose(0,2,3,1) # N,H,W,3
            pred_images = pred_images.transpose(0,2,3,1) # X,H,W,3
            ssim = calculate_ssim_batch(gt_images, pred_images, border=0)
            final_score.append(ssim)
        end = time.time()
        final_score = np.array(final_score).mean()

        print('-' * 100)
        print('time spent:',end-start,'s')
        print('gt_root_path:', gt_root_path)
        print('pred_root_path:', pred_root_path)
        print('view_num:', self.view_num)
        print('rendered_img_res:', self.rendered_img_res)
        print('-' * 100)
        print('SSIM', final_score)
        print('-' * 100)
        self.ssim = final_score


def vis_dataset():
    eval_test = RenderedImageDataset(gt_root_path=gt_root_path, pred_root_path=pred_root_path)
    data_loader = torch.utils.data.DataLoader(eval_test, batch_size=1, num_workers=1, \
                                              pin_memory=True, drop_last=False, shuffle=False)
    for i, data in enumerate(data_loader):
        gt_images,pred_images = data # [batch_size, view_num,3, res,res]

        for view_i in range(eval_test.view_num):
            cat = cat_images(gt_images[i][view_i],pred_images[i][view_i])
            display_CHW_RGB_img_np_matplotlib(cat)

def temp_get_some_samples(names_here):

    cls_id = '03001627'

    for name in names_here:
        path = os.path.join(gt_root_path,cls_id,name)
        if os.path.exists(path):
            if len(os.listdir(path)) == 40:
                print('nice',name)
            else:
                print('bad',name)
        else:
            print('bad',name)



def check_black_points():
    shapenet_sampled_pc_root_path = 'process_data/depth_renderer/datasets/pc'
    cls_id = '03001627'
    names = os.listdir(os.path.join(shapenet_sampled_pc_root_path,cls_id))
    has_black_point_num = 0
    from utils.vtk_basic import vis_actors_vtk,get_colorful_pc_actor_vtk
    for name in names:
        # name = 'f46ccdbf92b738e64b3c42e318f3affc'
        pc_file_dir = os.path.join(shapenet_sampled_pc_root_path, cls_id, name)
        # coords = np.load(os.path.join(pc_file_dir, 'coords.npy'))
        colors = np.load(os.path.join(pc_file_dir, 'colors.npy'))
        # coords = torch.tensor(coords, dtype=torch.float)
        colors = torch.tensor(colors, dtype=torch.float)
        zeros_tensor = torch.zeros_like(colors)
        equal_rows = torch.all(torch.eq(colors, zeros_tensor), dim=1)


def eval(dataset_name,pred_root_path):
    cls_id = os.listdir(pred_root_path)[0]
    if dataset_name == 'shapenet_core_v2':

        gt_root_path = 'datasets/shapenet_core_v2/rendered_imgs'

    elif dataset_name == 'google_scanned_objects':
        gt_root_path = 'datasets/google_scanned_objects/rendered_imgs'
    elif dataset_name == 'omniobject3d':
        gt_root_path = 'datasets/omniobject3d/rendered_imgs'

    print('pred_root_path', pred_root_path)
    print('gt_root_path', gt_root_path)


    # names_here = os.listdir(os.path.join(pred_root_path, cls_id))
    tester = Tester(cls_id=cls_id,pred_root_path=pred_root_path,gt_root_path=gt_root_path,dataset_name=dataset_name)

    tester.eval_lpips(gt_root_path=gt_root_path, pred_root_path=pred_root_path)
    tester.eval_psnr(gt_root_path=gt_root_path, pred_root_path=pred_root_path)
    tester.eval_ssim(gt_root_path=gt_root_path, pred_root_path=pred_root_path)
    tester.eval_fid(gt_root_path=gt_root_path, pred_root_path=pred_root_path)

    print('-----------------------------------------')
    print('pred_root_path',pred_root_path)
    print('gt_root_path',gt_root_path)
    print('sample num',len(tester.eval_set))
    print('fid\tlpips\tpsnr\tssim')
    print( tester.fid,'\t', tester.lpips,'\t', tester.psnr,'\t', tester.ssim,'\t')

    print('-----------------------------------------')
    now = time.strftime("%Y_%m_%d %H.%M.%S\n", time.localtime())
    with open(os.path.join(os.path.dirname(pred_root_path),f'{now}_eval_result.txt'), "a", encoding="utf-8") as f:
        # file_obj.write(scene_path)
        str = ''
        str += now
        str += f'pred root path: {pred_root_path}\n'
        str += f'GT root path: {gt_root_path}\n'
        str += f'sample num: {len(tester.eval_set)}\n'
        str += 'fid\tlpips\tpsnr\tssim\n'
        str += f'{tester.fid}\t{tester.lpips}\t{tester.psnr}\t{tester.ssim}\t'
        f.write(str)

        print()
def main():
    parser = argparse.ArgumentParser("run evaluation")
    parser.add_argument("--pred_root_path", type=str, default='', help="")
    parser.add_argument("--dataset_name", type=str, default='', help="") # shapenet_core_v2,google_scanned_objects,omniobject3d
    args = parser.parse_args()
    pred_root_path = args.pred_root_path
    dataset_name = args.dataset_name

    cls_id = os.listdir(pred_root_path)[0]
    eval(dataset_name,pred_root_path)

  


if __name__ == "__main__":
    main()