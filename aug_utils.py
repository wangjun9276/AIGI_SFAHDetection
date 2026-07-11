import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from io import BytesIO
import imageio
# import tensorflow as tf
import random


def plot_average_epoch_loss(loss_arr, n_epochs, exp_identifier, show=True):

    loss_arr = np.array(loss_arr)
    iter_epochs = loss_arr.shape[0] // n_epochs
    me = []
    for i in range(n_epochs):
        ls = loss_arr[i * iter_epochs:(i + 1) * iter_epochs]
        me.append(np.mean(ls))

    if show:
        plt.figure(figsize=(12, 9))
        plt.plot(np.arange(n_epochs), np.array(me), 'o-', markersize=8)
        plt.xticks(np.arange(0, n_epochs, step=1))
        plt.yticks(np.arange(0, np.max(me) + 0.5, step=0.5))
        plt.xlabel('Epoch ({:d} iterations)'.format(iter_epochs))
        plt.ylabel('Average loss')
        plt.title(exp_identifier)
        plt.grid()
        plt.savefig(f"{exp_identifier}.png")
        plt.close()
    else:
        plt.show()
        plt.close()

    return me


def augment(x, jpeg_prob, blur_prob, flip_prob, rotation_prob, scale_prob, noise_probability=0, cb_adjust=0, erase_prob=0, fname=''):
    """
    Random augmentations
    :param x: input image
    :return: augmented image
    """

    do_jpeg = np.random.randint(1, 101)
    do_rotate = np.random.randint(1, 101)
    do_flip = np.random.randint(1, 101)
    do_blur = np.random.randint(1, 101)
    do_gaussian_noise = np.random.randint(1, 101)
    do_scale = np.random.randint(1, 101)
    do_cb_adjust = np.random.randint(1, 101)
    do_erase = np.random.randint(1, 101)

    # Random JPEG compression
    if do_jpeg <= jpeg_prob:
        x = random_jpeg_compression(x)

    # Random rotation
    if do_rotate <= rotation_prob:
        x = random_rotate(x)

    # Random flip
    elif do_flip <= flip_prob:
        x = random_flip(x)

    elif do_gaussian_noise <= noise_probability:
        # if 'print_scan' not in fname:
        x = random_noise(x)

    # gaussian blur
    elif do_blur <= blur_prob:
        # kernels = [3]
        kernels = [3, 5, 7, 9]
        kernels_size = kernels[np.random.randint(len(kernels))]
        x = random_blur(x, kernels_size)

    # scaling
    elif do_scale <= scale_prob:
        x = random_scale(x)

    elif do_cb_adjust<=cb_adjust:
        b = np.random.randint(0, 5)
        c = np.random.randint(0, 5)
        x = ajust_image(x, cont=c, bright=b)

    elif do_erase <= erase_prob:
        x = RSM_DataGenerator(x, im_w=x.shape[1], im_h=x.shape[0], divide_w=10, divide_h=6, lowerlimit=0.7, upperlimit=0.7, postfix='jpg')

    return x

def random_noise(x):
    gauss = np.random.normal(0, 1, x.size)
    gauss = gauss.reshape((x.shape[0], x.shape[1], x.shape[2])).astype('uint8')
    return cv2.add(x, gauss)

def random_jpeg_compression(x):
    """
    JPEG compression in file stream with random quality
    :param x: input image
    :return: JPEG compressed image
    """
    qf = np.random.randint(40, 99)
    return np.array(jpeg_compression_in_buffer(x, qf))

def jpeg_compression_in_buffer(x_in, jpg_quality):

    x = Image.fromarray(np.uint8(x_in))

    buf = BytesIO()
    x.save(buf, format='jpeg', quality=jpg_quality)

    with BytesIO(buf.getvalue()) as stream:
        x_jpg = imageio.imread(stream)

    return x_jpg

def random_flip(img):
    '''
    random image flip
    Args:
        img:

    Returns:
        flipped images:
        0: vertical
        1: horizontal
        -1: rotation with angle 180
    '''
    rand = np.random.randint(-1, 2)
    return cv2.flip(img, rand)

def add_gasuss_noise(image, mean=0, var=0.001):
    '''
    gaussian noise processing
    Args:
        image:
        mean:
        var:

    Returns:

    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def random_blur(img, kernel):
    '''
    Gaussian blur
    Args:
        img:
        kernel:

    Returns:

    '''
    return cv2.GaussianBlur(img, (kernel, kernel), 0)

def random_scale(img):
    '''
    Gaussian blur
    Args:
        img:
        kernel:

    Returns:

    '''
    rows, cols, ch = img.shape
    scaler = [0.8, 0.9, 1, 1.1, 1.2, 1.3]
    index = np.random.randint(0, len(scaler))
    img = cv2.resize(img, (int(scaler[index]*rows), int(scaler[index]*cols)), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_CUBIC)

def random_rotate(img):
    ''' image rotation
    Args:
        img: original images

    Returns:
        img: rotated image
    '''
    rows, cols, ch = img.shape
    rand = np.random.randint(-45, 45)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rand, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img

def ajust_image(image, cont=1, bright=0):
    '''
        adjust contrast and brightness
        cont : contrast，modulated with brightness simultaneously
        bright : brightness
    '''
    out = np.uint8(np.clip((cont * image + bright), 0, 255))
    return out

def ajust_image_hsv(image, h=1, s=1, v=0.8):
    '''
        HSV channel adjustment
        channel parameters
    '''
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    H2 = np.uint8(H * h)
    S2 = np.uint8(S * s)
    V2 = np.uint8(V * v)
    hsv_image = cv2.merge([H2, S2, V2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return out

def ajust_jpg_quality(image, q=100, save_path=None):
    '''
        JPEG compression
        q : quality factor 0~100
    '''
    if save_path is None:
        cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        out = cv2.imread('jpg_tmp.jpg')
        return out
    else:
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), q])

def add_gasuss_blur(image, kernel_size=(3, 3), sigma=0.1):
    '''
        gaussian blur
        kernel_size : kernel size
        sigma : standard deviation
    '''
    out = cv2.GaussianBlur(image, kernel_size, sigma)
    return out

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = abs(min(width - x1, size_bb))
    size_bb = abs(min(height - y1, size_bb))

    return x1, y1, size_bb

def RSM_DataGenerator(path, im_w=66, im_h=100, im_channels=3, divide_w=6, divide_h=10, lowerlimit=0.97, upperlimit=1.0,
                      postfix='bmp'):
    alpha = random.uniform(lowerlimit, upperlimit)
    # print('random_ratio: ', alpha)
    # print('N: ', divide * divide)
    part = int(divide_h * divide_w * alpha)
    index = np.arange(0, divide_h * divide_w)
    # print(index)
    random.shuffle(index)
    # print(index)

    kb = index[0:part]
    kb = np.sort(kb)
    # kb=np.array([0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])

    # image = cv2.imread(path)
    image = path
    image = tf.image.resize(image, [im_h, im_w])
    # image = tf.cast(image, dtype=tf.float32)
    # print(type(image),image)
    # shape=image.get_shape().as_list()
    # print(shape)
    img_shape = image.shape
    print(img_shape, type(img_shape))
    rows = img_shape[0]
    cols = img_shape[1]
    w_cellsize = int(cols / divide_w)
    h_cellsize = int(rows / divide_h)
    img_list_h = []
    for i in range(0, divide_h):
        img_list_w = []
        for j in range(0, divide_w):
            offset_height = i * h_cellsize
            offset_width = j * w_cellsize
            target_height = h_cellsize
            target_width = w_cellsize
            img_temp = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

            if i * divide_h + j in kb:
                # print(i * divide + j, offset_height, offset_width, target_height, target_width)
                img_list_w.append(img_temp / 255.0)
            else:
                img_list_w.append(img_temp * 0)
        img_concat_w = tf.concat([im for im in np.array(img_list_w)], axis=1)
        img_list_h.append(img_concat_w)

    img_concat = tf.concat([im for im in np.array(img_list_h)], axis=0)
    return np.array(img_concat * 255, dtype='uint8')[:, :, [2, 1, 0]]

if __name__ == '__main__':
    # rand = np.randint(3)
    x = np.load('/home/benedetta/PycharmProjects/EyeGanDetectionSiamese/models/siamese/SiameseResNet50_lr5_128_ep5-66x100x3/losses_SiameseResNet50_lr5_128_ep5-66x100x3.h5.npy')
    epochs = 5
    plot_average_epoch_loss(x, epochs, 'figure_title', True)