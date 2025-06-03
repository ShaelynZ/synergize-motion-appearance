import numpy as np
from imageio import mimread, imread, mimsave
import warnings
import os

from basicsr.utils.matlab_functions import bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def frames2array(file, is_video, image_shape=None, column=0):
    if is_video:
        if os.path.isdir(file):
            images = [imread(os.path.join(file, name))  for name in sorted(os.listdir(file))]
            video = np.array(images)
        elif file.endswith('.png') or file.endswith('.jpg'):
            ### Frames is stacked (e.g taichi ground truth)
            image = imread(file)
            if image.shape[2] == 4:
                image = image[..., :3]

            video = np.moveaxis(image, 1, 0)
#            print (image_shape)
            video = video.reshape((-1, ) + image_shape + (3, ))
            video = np.moveaxis(video, 1, 2)
        elif file.endswith('.gif') or file.endswith('.mp4'):
            video = np.array(mimread(file))
        else:
            warnings.warn("Unknown file extensions  %s" % file, Warning)
            return []
    else:
        ## Image is given, interpret it as video with one frame
        image = imread(file)
        if image.shape[2] == 4:
            image = image[..., :3]
        video = image[np.newaxis]

    if image_shape is None:
        return video
    else:
        ### Several images stacked together select one based on column number
        return video[:, :, (image_shape[1] * column):(image_shape[1] * (column + 1))]