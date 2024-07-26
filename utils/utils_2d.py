##################################################################
#
#    Deal with 2D images as numpy arrays of shape (C,H,W)
#    C = 3, indicating RGB.
##################################################################
import os
import torch
import numpy as np
from torchvision.transforms import transforms
import cv2
import matplotlib.pyplot as plt
import PIL
from scipy.interpolate import griddata
import torch.nn.functional as F

def draw_hist(np_sequence):
    # Calculate the histogram
    hist, bins = np.histogram(np_sequence, bins=20)

    # Plot the histogram
    plt.hist(np_sequence, bins=bins)
    plt.show()

def frame_image(img,width = 2,color = (0,0,0)):
    '''
    frame the images by a frame with given width and color
    :param img: input image as numpy array of shape [C,H,W], where C is RGB, and range from 0 to 1
    :param width:
    :return:
    '''
    # Convert color tuple to numpy array
    color = np.array(color, dtype=np.float32)

    # Calculate the dimensions of the framed image
    h, w = img.shape[1:]
    fh, fw = h + 2 * width, w + 2 * width

    # Create a new numpy array for the framed image
    framed_img = np.zeros((3, fh, fw), dtype=np.float32)

    # Set the color of the frame
    # framed_img[:, :width, :] = color[:, np.newaxis] # left: [3,width, fw]; right: [3,1] # ValueError: could not broadcast input array from shape (3,1) into shape (3,2,260)
    # framed_img[:, -width:, :] = color[:, np.newaxis]
    # framed_img[:, width:-width, :width] = color[:, np.newaxis]
    # framed_img[:, width:-width, -width:] = color[:, np.newaxis]
    framed_img = framed_img.transpose((1,2,0)) # makd it H,W,C
    framed_img[:width, :, :] = color
    framed_img[-width:, :,:] = color
    framed_img[:, :width, :] = color
    framed_img[:, -width:, :] = color

    framed_img = framed_img.transpose((2,0,1))  # makd it back to C,H,W


    # Copy the input image into the framed image
    framed_img[:, width:h+width, width:w+width] = img

    return framed_img


def frame_img_cv2(image,x,y,w,h,color = (0,0,255),thickness = 2):
    '''
    in a given image, draw a rectangle inside a sub region of this image given by xywh.
    :param image: numpy, CHW, RGB
    :return:
    '''
    # Draw a rectangle around  the original image
    framed_image_cv2 = image.transpose(1, 2, 0)  # to H,W,C, still RGB
    framed_image_cv2 = cv2.cvtColor(framed_image_cv2, cv2.COLOR_RGB2BGR)  # to BGR
    # print(framed_image_cv2.shape, framed_image_cv2.dtype, )
    # #
    cv2.rectangle(framed_image_cv2, (x,y), (x+w, y+h), color=color, thickness=thickness)
    framed_image = cv2.cvtColor(framed_image_cv2, cv2.COLOR_BGR2RGB)  # to RGB, still HWC
    framed_image = framed_image.transpose(2, 0, 1)
    return framed_image

def resize_img_np(image, w,h):
    '''
    :param image: np, CHW,RGB
    :param w:
    :param h:
    :return:
    '''
    image = torch.tensor(image)
    image = transforms.Resize((h, w))(image)
    image = image.numpy()

    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.resize(image, (w, h))
    # image = np.transpose(image, (2, 0, 1))
    return image


def cat_images(img1, img2, margin=10, horizon = True):
    '''
    cat the img1 and img horizontally if horizon=True, else vertically.
    Resize img2 to fit the width of height of img1 using transforms.Resize() of pytorch like transforms.Resize((w, h))(img2)
    Both input images are np arrays of shape [C,H,W], RGB, ranging from 0 to 1. so as the result.

    :param img1:
    :param img2:
    :param margin:
    :param horizon:
    :return:
    '''
    # Convert numpy arrays to PyTorch tensors
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)


    # Resize img2 to match the dimensions of img1
    _,h1,w1 = img1.shape
    _,h2,w2 = img2.shape
    if horizon:
        img2 = transforms.Resize((h1, int(w2 * h1 / h2)))(img2)
        _, h2, w2 = img2.shape
        width = w1 + margin + w2
        height = h1
    else:
        img2 = transforms.Resize((int(h2 * w1 / w2), w1))(img2)
        _, h2, w2 = img2.shape
        width = w1
        height = h1 + margin + h2

    # Create an empty numpy array to hold the concatenated image
    concat_img = np.ones((img1.shape[0], height, width))

    # print('img1.shape',img1.shape)
    # print('img2.shape',img2.shape)
    # print('concat_img.shape',concat_img.shape)

    # Copy img1 into the left or top side of the concatenated image
    if horizon:
        concat_img[:, :img1.shape[1], :img1.shape[2]] = img1
    else:
        concat_img[:, :img1.shape[1], :img1.shape[2]] = img1

    # Copy img2 into the right or bottom side of the concatenated image
    if horizon:
        concat_img[:, :img2.shape[1], img1.shape[2] + margin:] = img2
    else:
        concat_img[:, img1.shape[1] + margin:, :img2.shape[2]] = img2

    return concat_img


def pad_and_cat_image(img1, img2, margin=10, horizon=True):
    '''
    Same as cat_images, but instead of resizing img2, padding it following:
    - If horizon==True and img2.h < img1.h, then padding img2 with white pixels on the top and bottom
      to make the padded img2 have the same height as img1.
    - If horizon==False and img2.w < img1.w, then padding img2 with white pixels on the left and right
      to make the padded img2 have the same width as img1.

    :param img1: np.array, the first image to concatenate
    :param img2: np.array, the second image to concatenate
    :param margin: int, the number of pixels between the two images
    :param horizon: bool, True to concatenate the images horizontally, False to concatenate vertically
    :return: np.array, the concatenated image
    '''
    # Convert numpy arrays to PyTorch tensors
    img1 = torch.tensor(img1)
    img2 = torch.tensor(img2)

    # Get the shapes of the input images
    _, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    # Pad img2 if necessary
    if horizon and h2 < h1:
        pad_top = (h1 - h2) // 2
        pad_bottom = h1 - h2 - pad_top
        img2 = np.pad(img2, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=1)
    elif not horizon and w2 < w1:
        pad_left = (w1 - w2) // 2
        pad_right = w1 - w2 - pad_left
        img2 = np.pad(img2, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant', constant_values=1)

    # Concatenate the images
    concat_img = cat_images(img1, img2, margin, horizon)

    return concat_img


def cat_image_sets(imgs1, imgs2, w, h, margin,by_row = False):
    '''
    Generate an image that has two columns (if by_row==False).
    The first columne are gt_imgs, and the second are pred_imgs
    each image need to be resized to width and height of w and h.

    :param imgs1: first set of images (np array of [C,H,W]). ranging from 0 and 1
    :param imgs2: second set of images (np array of [C,H,W])
    :param w:
    :param h:
    :param margin: margins between images
    :return:
    '''

    imgs1 = torch.tensor(imgs1)
    imgs2 = torch.tensor(imgs2)

    imgs1 = [transforms.Resize((w, h))(img) for img in imgs1]
    imgs2 = [transforms.Resize((w, h))(img) for img in imgs2]
    # mat_image = transforms.Resize((w, h))(img)  # [3,mat_img_size,mat_img_size]


    # Determine number of rows and columns
    if by_row:
        n_rows = len(imgs1) + len(imgs2)
        n_cols = 1
    else:
        n_rows = max(len(imgs1), len(imgs2))
        n_cols = 2

    # Create blank image to store concatenated image
    if len(imgs1) > 0:
        img_shape = imgs1[0].shape
    else:
        img_shape = imgs2[0].shape
    concat_img = np.ones(
        (img_shape[0], n_rows * (img_shape[1] + margin) - margin, n_cols * (img_shape[2] + margin) - margin))

    # Fill in concatenated image
    row_idx = 0
    for i, img1 in enumerate(imgs1):
        col_idx = 0
        concat_img[:, row_idx:row_idx + img_shape[1], col_idx:col_idx + img_shape[2]] = img1
        row_idx += img_shape[1] + margin
    row_idx = 0
    for i, img2 in enumerate(imgs2):
        col_idx = img_shape[2] + margin
        concat_img[:, row_idx:row_idx + img_shape[1], col_idx:col_idx + img_shape[2]] = img2
        row_idx += img_shape[1] + margin
    # if concat_img.max()<=1:
    #     concat_img = concat_img * 255
    #     concat_img = concat_img.astype(np.uint8)
    # print('concat_img',concat_img)
    return concat_img


def plot_sequence(data, w=6, h=3,title = 'Data Sequence'):
    """
    Plots a line chart for a sequence of 3-channel data using Matplotlib.
    The three channels are displayed in red, green, and blue respectively.

    :param data: sequence of 3-channel data of shape [L, 3]
    :param w: chart width
    :param h: chart height
    :return: chart as a numpy array of shape [H, W, 3]
    """

    import matplotlib.pyplot as plt


    # pre-process data
    # valid_len = (data[:, 0] > -100).sum()
    # data = data[:valid_len]
    data = data[data[:,0]>-100]

    # Create a new figure with the given width and height
    fig = plt.figure(figsize=(w, h))

    # Get the three channels of the data
    red = data[:, 0]
    if data.shape[1] == 3:
        green = data[:, 1]
        blue = data[:, 2]



    # Create a new axis object for the chart
    ax = fig.add_subplot(1, 1, 1)

    # Plot the three channels as three separate lines
    ax.plot(red, color='red')
    if data.shape[1] == 3:
        ax.plot(green, color='green')
        ax.plot(blue, color='blue')

    # # Set the x-axis and y-axis labels
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Channel Values')

    # Set the title of the chart
    ax.set_title(title)

    # Add the values above each point in the chart
    for i in range(len(data)):
        ax.text(i, red[i], str(round(red[i].item(), 6)), ha='center', va='bottom', fontsize=8)
        if data.shape[1] == 3:
            ax.text(i, green[i], str(round(green[i].item(), 6)), ha='center', va='bottom', fontsize=8)
            ax.text(i, blue[i], str(round(blue[i].item(), 6)), ha='center', va='bottom', fontsize=8)

    '''chatgpt'''
    # # Create a new canvas for the figure
    # from matplotlib.backends.backend_agg import FigureCanvasAgg
    # canvas = FigureCanvasAgg(fig)
    # # Render the figure to a numpy array
    # canvas.draw()
    # buf = canvas.buffer_rgba()
    # chart = np.asarray(buf)
    # # Close the figure to free up memory
    # plt.close(fig)
    # chart = chart[:,:, :3].transpose(2,0,1) / 255
    # return chart
    '''https://www.cnpython.com/qa/40418'''
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[:, :, :3].transpose(2, 0, 1) / 255 # make it C,H,W
    return data


def generate_image(color,w,h):
    '''
    given a color or (r,g,b), construct an image as a numpy array of shape [3,H,W], all piexels has the same color as input color
    :param color:
    :param w:
    :param h:
    :return:
    '''
    # Construct the image
    img = np.zeros((h, w, 3))
    img[:, :] = color

    # Return image
    return img.transpose((2, 0, 1))

def display_CHW_RGB_img_np_cv2(img):
    if isinstance(img,torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # from C,H,W to H,W,C
    pc_img = img[:, :, ::-1]  # from RGB to BGR
    cv2.imshow('img', pc_img)
    key = cv2.waitKey(0)  # Wait for keyboard input
    if key == 27:  # Check if Escape key was pressed
        cv2.destroyAllWindows()

def display_CHW_RGB_img_np_matplotlib(img):
    img = img.transpose(1, 2, 0)  # from C,H,W to H,W,C
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def display_CHW_RGB_img_np(img):
    display_CHW_RGB_img_np_cv2(img)

def save_CHW_RGB_img(img,file_name):
    img=img.transpose(1,2,0) #  # from C,H,W to H,W,C
    mask = (img==0).astype(np.uint8)
    img*=255 # from [0,1] to [0,255]
    img = img.clip(0, 255).astype(np.uint8)
    # img *= mask
    # img = np.ascontiguousarray(img[::-1, :, :])
    img = np.ascontiguousarray(img)
    # PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
    #     os.path.join(save_root, f'{name.split("/")[1]}.png'))


    image_pil = PIL.Image.fromarray(img,'RGB')

    image_pil.save(file_name)

def save_CHW_RGBA_img(img,file_name):
    img=img.transpose(1,2,0) #  # from C,H,W to H,W,C
    mask = (img==0).astype(np.uint8)
    img*=255 # from [0,1] to [0,255]
    img = img.clip(0, 255).astype(np.uint8)
    # img *= mask
    # img = np.ascontiguousarray(img[::-1, :, :])
    img = np.ascontiguousarray(img)
    # PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
    #     os.path.join(save_root, f'{name.split("/")[1]}.png'))


    image_pil = PIL.Image.fromarray(img,'RGBA')

    image_pil.save(file_name)

def load_CHW_RGB_img(file_name):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    inpainted_img = PIL.Image.open(file_name)
    if inpainted_img.mode != 'RGB':
        inpainted_img = inpainted_img.convert('RGB')  # this is important
    inpainted_img = torch.from_numpy(np.array(inpainted_img))  # H,W,C = 4
    inpainted_img = inpainted_img[:, :, :3]  # H,W,C = 3
    inpainted_img = inpainted_img.float() /255.
    # inpainted_img = torch.flip(inpainted_img, dims=(1,))
    # inpainted_img = inpainted_img.resize((res, res))
    inpainted_img = inpainted_img.permute(2,0,1)
    return inpainted_img

def load_CHW_RGBA_img_np(file_name,H=None,W=None):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    inpainted_img = PIL.Image.open(file_name)
    # print('inpainted_img.mode',inpainted_img.mode)
    assert inpainted_img.mode == 'RGBA'
    # if inpainted_img.mode != 'RGB':
    #     inpainted_img = inpainted_img.convert('RGB')  # this is important

    if H is not None and W is not None:
        inpainted_img = inpainted_img.resize((H,W))
    inpainted_img = np.array(inpainted_img)  # H,W,C = 4
    foreground_mask = inpainted_img[:,:,3:] # H,W,C = 1
    inpainted_img = inpainted_img[:, :, :3]  # H,W,C = 3
    inpainted_img = inpainted_img.astype(np.float32)/255.

    inpainted_img = inpainted_img.transpose(2,0,1) # CHW
    foreground_mask = foreground_mask.transpose(2,0,1)[0] # HW
    return inpainted_img,foreground_mask

def load_CHW_RGBA_img(file_name):
    '''
    CHW,RGB, float, 0-1
    :param file_name:
    :return:
    '''

    inpainted_img = PIL.Image.open(file_name)
    # print('inpainted_img.mode',inpainted_img.mode)
    assert inpainted_img.mode == 'RGBA'
    # if inpainted_img.mode != 'RGB':
    #     inpainted_img = inpainted_img.convert('RGB')  # this is important

    inpainted_img = torch.from_numpy(np.array(inpainted_img)).float() /255.  # H,W,C = 4
    foreground_mask = inpainted_img[:,:,3:]
    inpainted_img = inpainted_img[:, :, :3]  # H,W,C = 3

    # inpainted_img = torch.flip(inpainted_img, dims=(1,))
    # inpainted_img = inpainted_img.resize((res, res))
    inpainted_img = inpainted_img.permute(2,0,1)
    foreground_mask = foreground_mask.permute(2,0,1)
    return inpainted_img,foreground_mask

def make_2d_grid(min = -0.5,max = 0.5,resolution = 224):
    '''
    :param min: min coordinates of the grid
    :param max:
    :param resolution:
    :return: p: a torch tensor on cpu of shape: resolution^3, 3
    '''
    bb_min = (min,) * 2
    bb_max = (max,) * 2
    shape = (resolution,) * 2

    size = shape[0] * shape[1]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])

    pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1).expand(*shape).contiguous().view(size)


    p = torch.stack([pxs, pys], dim=1)

    return p


######################## numpy
def paint_pixels(img, pixel_coords, pixel_colors, point_size):
    '''
    :param img: numpy array of shape [3,res,res]
    :param pixel_coords: [N,2], 2 for H and W
    :param pixel_colors: [N,3]
    :param point_size: paint not only the given pixels, but for each pixel, paint its neighbors whose distance to it is smaller than (point_size-1).
    :return:
    '''
    N = pixel_coords.shape[0]
    C = img.shape[0]
    # print('img.shape',img.shape)
    # print('pixel_coords.shape',pixel_coords.shape)
    # print('pixel_colors.shape',pixel_colors.shape)
    if point_size == 1:
        img[:, pixel_coords[:, 0], pixel_coords[:, 1]] = pixel_colors.T
    else:
        pixel_coords = np.round(pixel_coords).astype(int)
        if point_size > 1:
            xx, yy = np.meshgrid(np.arange(-point_size + 1, point_size, 1), np.arange(-point_size + 1, point_size, 1))
            grid = np.stack((xx, yy), 2).reshape(point_size * 2 - 1, point_size * 2 - 1, 2) # grid_res,grid_res,2
            grid_res = grid.shape[0]
            grid = grid + pixel_coords.reshape(N, 1, 1, 2) # [N,grid_res,grid_res,2]
            pixel_colors = np.repeat(pixel_colors[:, np.newaxis, np.newaxis, :], grid_res, axis=1)  # [N,3] -> [N,grid_res,1,3]
            pixel_colors = np.repeat(pixel_colors[:, :, :, :], grid_res, axis=2)  # [N,3] -> [N,grid_res,grid_res,3]
            mask = (grid[:, :, :, 0] >= 0) & (grid[:, :, :, 0] < img.shape[1]) & \
                   (grid[:, :, :, 1] >= 0) & (grid[:, :, :, 1] < img.shape[2])  # [N,grid_res,grid_res],
            grid = grid[mask]
            # print('pixel_colors.shape',pixel_colors.shape)
            # print('mask.shape',mask.shape)
            pixel_colors = pixel_colors[mask]
            indices = grid.astype(int)
            img[:, indices[:,  0], indices[:, 1]] = pixel_colors.transpose((1,  0))

    return img

def fill_hole(binary_img,kernel_size = 7):
    '''
    :param binary_img: [H,W]
    :return:  [H,W]
    '''
    ''' by contours'''
    # contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # out_img = np.zeros((binary_img.shape[0], binary_img.shape[1]))
    #
    # for i in range(len(contours)):
    #     cnt = contours[i]
    #     cv2.fillPoly(out_img, [cnt], color=255)
    # return out_img
    '''morphology  close'''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return out_img

def naive_inpainting(img,mask2,method='linear'):
    '''
    Numpy
    :param img: C,H,W
    :param mask2: H,W. The background is True, foreground non-empty is True, foreground empty is False (to be painted)
    :param method: 'linear' or 'nearest'
    :return: C,H,W
    '''

    res = img.shape[1]

    # mask2 = mask2[0]
    need_to_fill_mask = ~(mask2.astype(np.bool_))

    # Create a grid of pixel coordinates
    y_coords, x_coords = np.indices(img.shape[1:])
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Flatten the image and mask arrays
    img_flat = img.reshape(img.shape[0], -1)
    mask_flat = need_to_fill_mask.ravel().astype(np.bool_)

    # Filter the image array to only include valid pixels
    valid_pixels = img_flat[:, ~mask_flat]
    valid_coords = coords[~mask_flat]

    x = np.arange(res)
    y = np.arange(res)
    xx, yy = np.meshgrid(x, y, indexing='xy')  # xy or ij

    # interpolated_pixels = griddata(valid_coords, valid_pixels.T, coords, input_pc_generate_method='nearest') # res*res,3
    # print('interpolated_pixels.shape', interpolated_pixels.shape)
    # filled_img = interpolated_pixels.transpose(1, 0)
    # filled_img = filled_img.reshape(-1, res, res)
    # # print('filled_img.shape',filled_img.shape)
    # save_CHW_RGB_img(filled_img, img_file_name)

    interpolated_pixels = griddata(valid_coords, valid_pixels.T, (xx, yy), method=method)  # res,res,3 # linear, nearest
    # print('interpolated_pixels.shape',interpolated_pixels.shape)
    filled_img = interpolated_pixels.transpose(2, 0, 1)
    # print('filled_img.shape',filled_img.shape)
    # save_CHW_RGB_img(filled_img, img_file_name)
    return filled_img

def smooth_img_researving_edges(img):
    '''
    :param img: res,res (gray img)
    :return:
    '''
    out_img = cv2.bilateralFilter(img.astype(np.float32), d=11, sigmaColor=10, sigmaSpace=150)
    out_img = cv2.bilateralFilter(out_img.astype(np.float32), d=11, sigmaColor=10, sigmaSpace=150)
    # src,d,sigmaColor,sigmaSpace,borderType
    # d: kernel size. size of neighbor
    # sigmaColor: only filter when |current pixel value - neighbor pixel value| smaller than this value
    # SigmaSpace: only work when d <=0. When d<=0, will use a d depending on sigmaSpace.
    return out_img

def detect_abnormal_bright_spots_in_gray_img(img,foreground_mask,save_path=None,
                                             min_for_norm=1.0,max_for_norm=3.0,
                                             edge_thresh = 50, pixel_num_thresh = 200,
                                             area_expand_thresh = 5,area_same_color_thres=5,
                                             brighter_thresh=6):
    '''
    :param img: [res,res],
    :param foreground_mask:
    :param min_for_norm: 1.0
    :param max_for_norm: 3.0
    :param edge_thresh: 50 # a bright spot should have an obvious edge
    :param pixel_num_thresh: 80 # a bright spot should have smaller area than thresh
    :param area_expand_thresh: 5
    :param area_same_color_thres: 5
    :param brighter_thresh: 6 # a bright spot should be brighter than pixels around it

    :return:
    '''
    res = img.shape[0]
    # print('img.shape',img.shape)
    # print('before img',img.max(),img.min())
    uint8_img = (img-min_for_norm) / (max_for_norm-min_for_norm)
    # print('after uint8_img', uint8_img.max(), uint8_img.min())
    uint8_img = uint8_img*255.0
    uint8_img = np.clip(uint8_img,0,255)
    uint8_img = uint8_img.astype(np.uint8)
    # detect edges:
    # print('uint8_img.max(),uint8_img.min()',uint8_img.max(),uint8_img.min())
    ### canny
    # edges = cv2.Canny(uint8_img, threshold1=30, threshold2=20)
    ### scharr
    im1x = cv2.Scharr(uint8_img, cv2.CV_64F, 1, 0)
    im1y = cv2.Scharr(uint8_img, cv2.CV_64F, 0, 1)
    im1x = cv2.convertScaleAbs(im1x)
    im1y = cv2.convertScaleAbs(im1y)
    edges = cv2.addWeighted(im1x, 0.5, im1y, 0.5, 0)

    src_img_color = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)



    edges_binary = edges.copy()
    edges_binary[edges>edge_thresh] = 0
    edges_binary[edges<=edge_thresh] = 255
    num_labels, labels = cv2.connectedComponents(edges_binary, connectivity=8)

    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    color_img = np.zeros((res, res, 3), dtype=np.uint8)



    dilate_kernel = np.ones((3, 3), np.uint8)

    abnormal_mask = np.zeros((res,res)).astype(np.bool_)

    # print('-----------------------------------------------')
    for i in range(num_labels):
        abnormal = False
        pixel_num = (labels==i).astype(np.int32).sum()
        # print('pixel_num',pixel_num)
        if pixel_num< pixel_num_thresh: # a bright spot should be smaller than threshold
            label_area_mask = labels == i
            dilated_label_area_mask = cv2.dilate(label_area_mask.astype(np.uint8), dilate_kernel,
                                                 iterations=area_expand_thresh).astype(np.bool_)
            # print('src_img_color[dilated_label_area_mask]',src_img_color[dilated_label_area_mask].max(),
            #       src_img_color[dilated_label_area_mask].min())
            background_pixel_num = np.logical_and(dilated_label_area_mask, np.logical_not(foreground_mask)).astype(np.int32).sum()
            if background_pixel_num<1:  # we only deal with bright spots in the foreground
                mean_color = uint8_img[label_area_mask].astype(np.float64).mean()
                same_color_mask = np.abs(uint8_img.astype(np.float64) - mean_color) < area_same_color_thres
                final_label_area_mask = np.logical_and(dilated_label_area_mask, same_color_mask)

                around_pixel_mask = np.logical_and(dilated_label_area_mask, np.logical_not(final_label_area_mask))
                around_pixel_mean_color = uint8_img[around_pixel_mask].mean()
                # print('---------------')
                # print('around_pixel_mean_color,mean_color',around_pixel_mean_color,mean_color)
                if (mean_color-around_pixel_mean_color) > brighter_thresh: # a bright spot should be brighter than pixels around
                    abnormal = True
                    abnormal_mask[final_label_area_mask] = True
                    # print('src_img_color[labels==i]',src_img_color[labels==i].max(),src_img_color[labels==i].min())

                    # print("--------------------------")

                    # color_img[labels==i] = colors[i]
                    color_img[final_label_area_mask] = colors[i]
                    # color_img[dilated_label_area_mask] = colors[i]

        if not abnormal:
            color_img[labels==i] = src_img_color[labels==i]


    if save_path is not None:
        depth_map_file = save_path
        # remove abnormal in depth
        depth_with_abnormal = src_img_color.transpose(2, 0, 1)[0][np.newaxis, ...] # [1,H,W]
        non_abnormal_value_mask = np.logical_not(abnormal_mask)
        dense_without_abnormal = naive_inpainting(img=depth_with_abnormal,  # [1,res,res]
                                 mask2=non_abnormal_value_mask[np.newaxis, ...],  # [1,res,res]
                                 method='nearest')[0]

        # # non_abnormal_value_mask = torch.logical_and(non_empty_pixel_mask, torch.logical_not(abnormal_masks[i]))
        # # dense = naive_inpainting(img=(sparse_depth_maps[i]).unsqueeze(0),  # [1,res,res]
        # #                          no_need_inpaint_mask2=non_abnormal_value_mask.unsqueeze(0),  # [1,res,res]
        # #                          method='nearest')[0]
        #
        # dense = (dense - min_for_norm) / (max_for_norm - min_for_norm)
        # dense[dense < 0] = 0
        dense_without_abnormal[~foreground_mask] = 0  # apply foreground_mask in depth [res,res]

        depth_map_img = np.repeat(dense_without_abnormal[np.newaxis, ...], 3, axis=0).astype(np.float64)  # [3,res,res]

        # prepare other images: original, edges, bright spots marked by colors, after
        src_img_color_with_edges = src_img_color.copy()
        src_img_color_with_edges[edges > edge_thresh] = (0, 0, 255)  #

        src_img_color = src_img_color * foreground_mask[..., np.newaxis]
        src_img_color_with_edges = src_img_color_with_edges * foreground_mask[..., np.newaxis]
        color_img = color_img * foreground_mask[..., np.newaxis]


        cat = cat_images(src_img_color.transpose(2, 0, 1), src_img_color_with_edges.transpose(2, 0, 1)) / 255.0
        cat = cat_images(cat, color_img.transpose(2, 0, 1) / 255.0)
        cat = cat_images(cat,depth_map_img/ 255.0)
        cat = np.flip(cat, 1)  # flip depth map upside down
        save_CHW_RGB_img(cat, depth_map_file)
    return abnormal_mask

    # # binary
    # thresh = cv2.adaptiveThreshold(uint8_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # return thresh
    #
    # # find contours: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    # contours, hierarchy = cv2.findContours(uint8_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     mask = np.zeros_like(img)
    #     cv2.drawContours(mask, [contour], -1, 255, -1)
    # return mask


def detect_edges_in_gray_by_scharr(gray_img_uint8):
    '''

    :param gray_img_uint8: numpy array, uint8, [res,res], ranging from 0 to 255
    :return:
    '''
    im1x = cv2.Scharr(gray_img_uint8, cv2.CV_64F, 1, 0)
    im1y = cv2.Scharr(gray_img_uint8, cv2.CV_64F, 0, 1)
    im1x = cv2.convertScaleAbs(im1x)
    im1y = cv2.convertScaleAbs(im1y)
    edges = cv2.addWeighted(im1x, 0.5, im1y, 0.5, 0)
    return edges






#################################### pytorch
@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # ���� kernelsize ����Ĭ�ϵ� sigma���� opencv ����һ��
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # Ԫ����������ĵĺ������
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # ����һά������
    # ����ָ���������ʣ����þ���˷����ټ����ά������
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # ��һ��
    return kernel

@torch.no_grad()
def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    '''
    :param batch_img:  BxcxHxW
    :param ksize:
    :param sigmaColor:
    :param sigmaSpace:
    :return:
    '''
    # https://www.jianshu.com/p/8c9e9e57d48e
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img ��ά��Ϊ BxcxHxW, ���Ҫ���ŵ� ������ά�� unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    # ����������Ȳ�
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # �����������Ȳ����Ȩ�ؾ���
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # ��һ��Ȩ�ؾ���
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # ��ȡ gaussian kernel �����临�Ƴɺ� weight_color ��״��ͬ�� tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # ����Ȩ�ؾ�����˵õ��ܵ�Ȩ�ؾ���
    weights = weights_space * weights_color
    # ��Ȩ�ؾ���Ĺ�һ������
    weights_sum = weights.sum(dim=(-1, -2))
    # ��Ȩƽ��
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix

def detect_edges_in_gray_by_scharr_torch_batch(gray_imgs_float32):
    '''
    :param gray_imgs_float32: torch tensor, [B,res,res], ranging from 0.0 to 255.0 (B indicates batch size)
    :return: edges: torch tensor, [B,res,res], ranging from 0.0 to 1.0 (B indicates batch size)
    '''
    device = gray_imgs_float32.device

    # Calculate the x and y gradients using the Scharr kernel
    # We use the Scharr kernel with ddepth = -1 to get the gradients in the same data type as the input
    kernel = torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float32).to(device)
    im1x = torch.nn.functional.conv2d(gray_imgs_float32, kernel, padding=1) # BCHW

    im1x =  torch.abs(im1x)

    kernel = torch.tensor([[[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float32).to(device)
    im1y = torch.nn.functional.conv2d(gray_imgs_float32, kernel, padding=1)
    im1y = torch.abs(im1y)


    # Compute the edge map by combining the x and y gradients with equal weights
    edges = torch.add(im1x, im1y) / 2.0

    # vis = True
    # if vis:
    #     cat = cat_images(im1x[0].clip(0,255).repeat(3,1,1).detach().cpu().numpy()/255.0,
    #                      im1y[0].clip(0, 255).repeat(3, 1, 1).detach().cpu().numpy()/255.0,
    #                      )
    #     display_CHW_RGB_img_np_matplotlib(cat)
    return edges





def dilate_torch_batch(binary_img_batch, kernel_size):
    """
    dilate white pixels like cv2.dilate
    :param img_batch: [B,  H, W] ,B indicate batch size. each element is either 0 or 1
    :param kernel_size:
    :return: [B,  H, W] same size as before
    """

    pad = (kernel_size - 1) // 2
    bin_img = F.pad(binary_img_batch.unsqueeze(1), pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=kernel_size, stride=1, padding=0)
    out = out.squeeze(1)
    return out




def dilate_foreground(N4HW_RGBA_imgs,iterations=2):
    '''
    '''
    device = N4HW_RGBA_imgs.device
    N4HW_RGBA_imgs = N4HW_RGBA_imgs*255.0
    N4HW_RGBA_imgs = N4HW_RGBA_imgs.long()
    dilated_imgs = torch.zeros_like(N4HW_RGBA_imgs).to(device).float()
    for i in range(len(N4HW_RGBA_imgs)):
        arr = N4HW_RGBA_imgs[i].permute(1,2,0).detach().cpu().numpy().astype(np.uint8) # 3HW to HW3
        forground_mask = N4HW_RGBA_imgs[i,-1].bool().detach().cpu().numpy() # .astype(np.uint8)  HW
  
        # dialte foreground
        background_mask = np.logical_not(forground_mask)

        kernel = np.ones((3,3), 'uint8')
        rgb = arr[:,:,:3]
        rgba=arr
        
        
        rgba[:,:,3] = forground_mask*255
        rgb[background_mask] =0
        for j in range(iterations):

            dilated_rgba = cv2.dilate(rgba, kernel,iterations=1)  #

            rgba = rgba * (1 - background_mask[...,np.newaxis]) + dilated_rgba * background_mask[...,np.newaxis]
            rgba = rgba.astype(np.uint8)


        dilated_imgs[i] = torch.tensor(rgba).permute(2,0,1).to(device).float()/255.0
    return dilated_imgs

    
    # kiui.vis.plot_image(rgba)
    
    out_img_list.append(img)
    return out_img_list

if __name__ == '__main__':

    def test_dilate_foreground():
        import sys
        import kiui
        sys.path.append('.')
        sys.path.append('..')
        device = torch.device('cuda')
        inpainted_img,foreground_mask = load_CHW_RGBA_img('output/clock_default/others/7_inpainted.png')
        kiui.lo(inpainted_img)
        kiui.lo(foreground_mask)
        CHW_RGBA = torch.cat([inpainted_img,foreground_mask],dim=0).to(device).float()
        # kiui.vis.plot_image(CHW_RGBA[:3].permute(1,2,0))
        NCHW_RGBA = CHW_RGBA.unsqueeze(0)
        dilated = dilate_foreground(NCHW_RGBA)[0]
        cat = cat_images(inpainted_img.detach().cpu().numpy(),dilated[:3].detach().cpu().numpy())
        
        kiui.vis.plot_image(cat.transpose(1,2,0))
    test_dilate_foreground()
        