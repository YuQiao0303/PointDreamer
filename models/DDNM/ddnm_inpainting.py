import sys
import time

sys.path.append('.')
sys.path.append('..')
sys.path.append('...')
import os
import yaml
from munch import Munch
import numpy as np
import torch
from models.DDNM.guided_diffusion.diffusion import Diffusion
from utils.utils_2d import display_CHW_RGB_img_np_matplotlib,cat_images,load_CHW_RGB_img

class Inpainter():
    def __init__(self, device):
        # get config
        with open(os.path.join("models/DDNM/configs/imagenet_256.yml"), "r") as f:
            config  = Munch.fromDict(yaml.safe_load(f))
        args = {
            "sigma_y":0,
            "eta": 0.85,
            "seed": 1234, # Set different seeds for diverse results
        }
        args = Munch.fromDict(args)
        self.runner = Diffusion(args, config,device=device)
        self.model = self.runner.get_model()

    def inpaint(self,masked_imgs,masks):
        '''
        :param masked_imgs: torch tensor, [3,H,W](RGB), float, 0-1. H=W=256
        :param masks: torch tensor, [H,W], float32
        :return: inpainted_img: torch on device, CHW, RGB
        '''

        # simplified_ddnm_inpainting
        # masked_imgs: 1,1,3,256,256
        # masks: 1,256,256
        inpainted_imgs = self.runner.simplified_ddnm_inpainting(self.model,
                          masked_imgs.permute(0,3,1,2).unsqueeze(0), # from [1,256,256,3] to [1,1,3,256,256]
                          masks[:,:,:,0]) # [1,1,3,256,256]
        inpainted_imgs = inpainted_imgs[0]

        return inpainted_imgs

import torch.multiprocessing as mp

def process_image(args):
    img_idx,inpainter, masked_img, mask = args
    return inpainter.inpaint(masked_img.unsqueeze(0), mask.unsqueeze(0))

def inpaint_batch(inpainter,masked_imgs,masks,cpu_core=4):
    ''' DON'T USE THIS, EVEN SLOWER
    :param masked_imgs: torch tensor, [N,3,H,W](RGB), float, 0-1. H=W=256
    :param masks: torch tensor, [N,H,W], float32
    :return: inpainted_img: torch on device, NCHW, RGB
    :return:
    '''

    mp.set_start_method('spawn', force=True)
    # pool = mp.Pool(mp.cpu_count()) # use all cpu cores
    pool = mp.Pool(cpu_core)
    try:
        results = pool.map(process_image, [(i,inpainter, img, mask)
                                       for i, (img, mask) in enumerate(zip(masked_imgs, masks))])
    finally:
        pool.close()
        pool.join()


    inpainted_imgs = torch.cat(results,0) # N,C,H,W
    # print('inpainted_imgs.shape',inpainted_imgs.shape)
    return inpainted_imgs


if __name__ == '__main__':

    device = torch.device('cuda')
    print('Loading model...')
    inpainter = Inpainter(device)
    print('Loading images')

    masked_img_path = 'output/clock/others/7_inpainted.png'
    masked_img = load_CHW_RGB_img(masked_img_path).to(device)
    

    prob = 0.8
    res=256
    all_masks = []
    batch_size = 4
    for i in range(batch_size):
        mask_vec = torch.ones([1, res * res])
        samples = np.random.choice(res * res, int(res ** 2 * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, res, res)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(masked_img, device=masked_img.device)
        mask[:, ...] = mask_b
        all_masks.append(mask.unsqueeze(0))
    all_masks = torch.cat(all_masks)  # batch_size,C,H,W

    masked_imgs = masked_img.permute(1, 2, 0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # batch_size,H,W,C
    all_masks = all_masks.permute(0, 2, 3, 1)  # batch_size,H,W,C

    print('Inpainting...')
    
    # batch_wise:
    start = time.time()
    inpainted_imgs = inpaint_batch(inpainter,masked_imgs,all_masks,cpu_core=batch_size)
    end = time.time()
    print('batch_wise, time: ', end-start,'s')


    start = time.time()
    for i in range(batch_size):
        start_i = time.time()
        inpainted_imgs = inpainter.inpaint(masked_imgs=masked_imgs[i].unsqueeze(0),
                                           masks=all_masks[i].unsqueeze(0))
        end_i = time.time()
        print(i,', time: ', end_i - start_i, 's')
    end = time.time()
    print('one by one total, time: ', end - start, 's')

