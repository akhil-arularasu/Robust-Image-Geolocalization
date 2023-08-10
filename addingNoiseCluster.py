#unused code

import os, cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
from dataset.CVUSA import CVUSA
import argparse
from scipy.ndimage import map_coordinates
import csv
from torchvision.transforms import ToPILImage


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
import wand
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image, ImageFilter
from scipy.ndimage import zoom as scizoom
#from scipy.ndimage.interpolation import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)
print('hello')
def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


from scipy.ndimage import zoom

def clipped_zoom(img, zoom_factor):
    h, w, _ = img.shape
    zoom_tuple = (zoom_factor,) * 2 + (1,)  # Keep the same zoom factor for all channels
    zh = int(np.round(h / zoom_factor))
    zw = int(np.round(w / zoom_factor))
    top = (h - zh) // 2
    left = (w - zw) // 2
    zoomed_img = zoom(img, zoom_tuple)
    result = np.zeros_like(img)
    result[top:top+zh, left:left+zw, :] = zoomed_img[top:top+zh, left:left+zw, :]
    return result


# /////////////// End Distortion Helpers ///////////////

# /////////////// Distortions ///////////////


def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def fgsm(x, source_net, severity=1):
    c = [8, 16, 32, 64, 128][severity - 1]

    x = V(x, requires_grad=True)
    logits = source_net(x)
    source_net.zero_grad()
    loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
    loss.backward()

    return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    height, width, _ = x.shape

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                if 0 <= h < height and 0 <= w < width and 0 <= h_prime < height and 0 <= w_prime < width:
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255

def defocus_blur(image_array, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    image_array = np.array(image_array)

    blurred_channels = []
    for d in range(3):
        blurred_channels.append(gaussian_filter(image_array[:, :, d], sigma=c[0]))
    blurred_channels = np.array(blurred_channels).transpose((1, 2, 0))  # 3xHxW -> HxWx3

    blurred_image_array = np.clip(blurred_channels, 0, 255).astype(np.uint8)
    blurred_image = Image.fromarray(blurred_image_array.astype(np.uint8))

    return blurred_image

def motion_blur(x, severity=1, image_number=0):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    # Convert x to a NumPy array if it is a PIL Image
    if isinstance(x, Image.Image):
        x_np = np.array(x)
    else:
        x_np = x

    # Apply motion blur using the PIL ImageFilter module
    x_blur = Image.fromarray(x_np).filter(ImageFilter.BLUR)

    # Rotate the blurred image
    angle = np.random.uniform(-45, 45)
    x_blur_rotated = x_blur.rotate(angle)

    # Convert the rotated image back to a NumPy array
    x_blur_rotated_np = np.array(x_blur_rotated)

    # Clip the pixel values between 0 and 255
    x_blur_rotated_np = np.clip(x_blur_rotated_np, 0, 255)

    return x_blur_rotated_np.astype(np.uint8)



def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        zoomed_x = clipped_zoom(x, zoom_factor)
        out += zoomed_x

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


# def barrel(x, severity=1):
#     c = [(0,0.03,0.03), (0.05,0.05,0.05), (0.1,0.1,0.1),
#          (0.2,0.2,0.2), (0.1,0.3,0.6)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#
#     x = WandImage(blob=output.getvalue())
#     x.distort('barrel', c)
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()

    # Resize plasma_fractal output to match the dimensions of x
    plasma = plasma_fractal(wibbledecay=c[1])
    resized_plasma = cv2.resize(plasma, (x.shape[1], x.shape[0]))

    x += c[0] * resized_plasma[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

'''
def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)
'''

def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = x.resize((int(224 * c), int(224 * c)), Image.BOX)
    x = x.resize((224, 224), Image.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////


# /////////////// Display Results ///////////////

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--sat_res', type=int, default=0, help='satellite image resolution')
parser.add_argument('--fov', type=int, default=0, help='field of view')
# Parse the command-line arguments
args = parser.parse_args()
args.mode = 'test_query'  # Set the mode attribute to
# Pass the args object to the CVUSA class
dataset = CVUSA(mode='test_query', args=args)
image_tuple = dataset[0]

print(len(dataset))

# Extract the image tensor from the tuple
image = image_tuple[0]
# image = image.reshape(image.shape[1], image.shape[2], image.shape[0])

'''
# Convert the image tensor to a NumPy array
image_array = image.numpy()
image_array = image_array.t((1, 2, 0))
image_array = image_array / 255.0  # Normalize the image data
image_array = image_array.clip(0, 1)  # Clip values to the valid range [0, 1]
'''

import collections

d = collections.OrderedDict()
d['Gaussian Noise'] = gaussian_noise
d['Shot Noise'] = shot_noise
d['Impulse Noise'] = impulse_noise
d['Defocus Blur'] = defocus_blur
d['Glass Blur'] = glass_blur
d['Motion Blur'] = motion_blur
d['Zoom Blur'] = zoom_blur
d['Snow'] = snow
## d['Frost'] = frost
d['Fog'] = fog
d['Brightness'] = brightness
d['Contrast'] = contrast
d['Elastic'] = elastic_transform
d['Pixelate'] = pixelate
d['JPEG'] = jpeg_compression

d['Speckle Noise'] = speckle_noise
d['Gaussian Blur'] = gaussian_blur
d['Spatter'] = spatter
d['Saturate'] = saturate

image_list = []
input_csv_path = '/home/c3-0/parthpk/CVUSA/splits/val-19zl.csv'
output_directory = '/home/ak362297/TransGeo2022/FINALCVUSANoiseSeverity1NEWNOISEBEFORETRANSFORM' # when changing noise severity change HERE and

with open(input_csv_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        image_path = row[1]  # Assuming the streetview/panos JPG image path is in the second column
        image_list.append(image_path)

image_list = [item.replace('streetview/panos/', '') for item in image_list]
print('lllen of image list')
print(len(image_list))
image_directory = '/home/c3-0/parthpk/CVUSA/streetview/panos'
#output directories for each noise type
for noise_name, noise in d.items():
    noise_folder = os.path.join(output_directory, noise_name)
    os.makedirs(noise_folder, exist_ok=True)

index = 0
for image_name in image_list:
    image_path = os.path.join(image_directory, image_name)

    # Open the image
    img = Image.open(image_path)
    # Apply noise functions to the image
    for noise_name, noise_func in d.items():
        severity = 1 # and change HERE

        noisy_img = noise_func(img, severity)

        # Convert the noisy image to a NumPy array and perform any necessary processing
        noisy_img_np = np.array(noisy_img)

        # Convert the data type to uint8
        noisy_img_np = noisy_img_np.astype(np.uint8)

        # Save the noisy image to the output directory
        output_path = os.path.join(output_directory, noise_name, image_name)
        Image.fromarray(noisy_img_np).save(output_path)

        print(f"Applied {noise_name} with severity level {severity} to {image_name} and saved to {output_path}")
        print(index)
        index = index+1

'''
if len(dataset) != len(image_list):
    print("Error: Mismatch between dataset length and image_list length.")
    print(f"Dataset length: {len(dataset)}, image_list length: {len(image_list)}")
else:
    for index in range(len(dataset)):
        # Get the data item at the current index
        img_query, img_reference, idx = dataset[index]
        image_name = image_list[index]
    
        for noise_name, noise_func in d.items():
            print(noise_name)
            noisy_img = noise_func(img_query)

            # Convert the noisy image to a NumPy array and perform multiplication
            noisy_img = np.array(noisy_img) * 255
            noisy_img = noisy_img.astype(np.uint8)

            # Transpose the dimensions of the array if necessary
            if noisy_img.shape[0] != 3:
                noisy_img = np.transpose(noisy_img, (1, 2, 0))


            # Save the noisy image to the corresponding output directory
        output_path = os.path.join(output_folder, noise_name, image_name)

        # Reshape the noisy_img array if it has a shape of (1, 1, 616)
        if len(noisy_img.shape) == 3 and noisy_img.shape[0] == 1 and noisy_img.shape[1] == 1:
            noisy_img = noisy_img.reshape(noisy_img.shape[2], noisy_img.shape[0], noisy_img.shape[1])

        # Convert the noisy_img array to the appropriate data type and save the image
        Image.fromarray(noisy_img.astype(np.uint8)).save(output_path)

        print(f"Applied {noise_name} to {image_name} and saved to {output_path}")
'''


'''
for image_name in image_list:
    image_path = os.path.join(image_directory, image_name)

    # Open the image
    img = Image.open(image_path)

    # Apply noise functions to the image
    for noise_name, noise_func in d.items():
        noisy_img = noise_func(img)

        # Convert the noisy image to a NumPy array and perform any necessary processing
        noisy_img_np = np.array(noisy_img)

        # Convert the data type to uint8
        noisy_img_np = noisy_img_np.astype(np.uint8)

        # Save the noisy image to the output directory
        output_path = os.path.join(output_directory, noise_name, image_name)
        Image.fromarray(noisy_img_np).save(output_path)

        print(f"Applied {noise_name} to {image_name} and saved to {output_path}")
'''
