import torch	
import torchvision.transforms as transforms	
from PIL import Image, ImageEnhance, ImageFilter	
import numpy as np	
import os	
import random	
import sys	
import cv2	
import torchvision.transforms.functional as functional	
from torchvision.transforms.functional import to_pil_image	
import trainingNoiseCluster


class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        if isinstance(x, int):
            # Handle the case when `x` is an integer
            return x
        # print(x.shape)
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x
        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
        transforms.Lambda(lambda x: x[:min([len(x)] + [len(y) for y in x])])

    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def add_gaussian_noise(img, mean=0.0, std=1.0):
    std = std * 0.05  # Adjust the standard deviation based on the severity level
    tensor_img = functional.to_tensor(img)
    noise = torch.normal(mean=mean, std=std, size=tensor_img.size())
    noisy_tensor_img = tensor_img + noise
    noisy_img = functional.to_pil_image(noisy_tensor_img)
    return noisy_img

# pytorch version of CVUSA loader
class CVUSA_Noise4Gaussian(torch.utils.data.Dataset):

    def __init__(self, mode = '', root = '/home/c3-0/parthpk/CVUSA/', query_root = '/home/ak362297/TransGeo2022/FINALCVUSANoiseSeverity4/',same_area=True, print_bool=False, polar = '', args=None, noiseName='Gaussian Blur', severity = 4): #CV-dataset
        super(CVUSA_Noise4Gaussian, self).__init__()

        self.noiseName = noiseName
        self.args = args
        self.severity = severity
        self.root = root
        self.query_root = query_root #unused
        self.polar = polar
        self.mode = mode
        self.sat_size = [256, 256]
        self.sat_size_default = [256, 256]
        self.grd_size = [112, 616]
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]
        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [750, 750]
        self.grd_ori_size = [224, 1232]
        
        if args.fov != 0:
            self.transform_query = input_transform_fov(size=self.grd_size,fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)

        if len(polar) == 0:
            self.transform_reference = input_transform(size=self.sat_size)
        else:
            self.transform_reference = input_transform(size=[750,750]) # 750, 512

        self.to_tensor = transforms.ToTensor()

        self.train_list = self.root + 'splits/train-19zl.csv'
        self.test_list2 = self.root + 'splits/val-19zl.csv'
        self.test_list = '/home/ak362297/TransGeo2022/CVUSANoise_CSVs/' + noiseName + '.csv' # desired directory for csv file

        test_refListlen = 0
        self.id_test_list2 = []
        self.id_test_list2idx = []
        with open(self.test_list2, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
             #  print('pano_id' + pano_id)
             #  satellite filename, streetview filename, pano_id
                self.id_test_list2.append([data[0], data[1], pano_id])
                self.id_test_list2idx.append(idx)
                idx += 1

        if print_bool:
            print('CVUSA: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_list.append([data[0], data[1], pano_id])
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        if print_bool:
            print('CVUSA: load', self.train_list, ' data_size =', self.data_size)
            print('CVUSA: load %s' % self.test_list2)
            print('CVUSA: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []

        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
                # satellite filename, streetview filename, pano_id
                self.id_test_list.append([data[0], data[1], pano_id])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        if print_bool:
            print('CVUSA: load', self.test_list, ' data_size =', self.test_data_size)

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open('/home/c3-0/parthpk/CVUSA/' + self.id_list[idx][1])
            # Convert image to RGB if it's not already in RGB format
            if img_query.mode != 'RGB':
                img_query = img_query.convert('RGB')
            img_reference = Image.open('/home/c3-0/parthpk/CVUSA/' + self.id_list[idx][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)

            # Randomly select a noise type
            #noise_type = random.choice(['Gaussian Noise', 'Shot Noise', 'Impulse Noise', 'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur', 'Snow', 'Fog', 'Brightness', 'Contrast', 'Elastic', 'Pixelate', 'JPEG', 'Speckle Noise', 'Gaussian Blur', 'Spatter', 'Saturate'])
            noise_type = 'Gaussian Noise'           	
            if noise_type == 'Gaussian Noise':	
                noisy_img_query = add_gaussian_noise(img_query, mean=0.0, std=self.severity)	
            if noise_type == 'Brightness':	
                brightness_factors = [1.2, 1.4, 1.6, 1.8, 2.0]
                enhancer = ImageEnhance.Brightness(img_query)	
                noisy_img_query = enhancer.enhance(brightness_factors[self.severity - 1])
         #       print('brightness', type(noisy_img_query))
            elif noise_type == 'Contrast':	
                contrast_factors = [0.4, 0.3, 0.2, 0.1, 0.05]
                contrast_factor = contrast_factors[self.severity - 1]
                noisy_img_query = functional.adjust_contrast(img_query, contrast_factor=contrast_factor)
           #     print('Contrast', type(noisy_img_query))
            elif noise_type == 'Gaussian Blur':	
                radius = [1, 2, 3, 4, 5][self.severity - 1]	
                noisy_img_query = img_query.filter(ImageFilter.GaussianBlur(radius))
           #     print('gaussian blur', type(noisy_img_query))
            elif noise_type == 'Hue':
                img_hue_factor  = [0.04, 0.08, 0.12, 0.16, 0.2][self.severity - 1]
                noisy_img_query = functional.adjust_hue(img_query, img_hue_factor)
           #     print('hue', type(noisy_img_query))	
            elif noise_type == 'Saturate':	
                saturation_factor = [0.5, 0.75, 1.0, 1.25, 1.5][self.severity - 1]	
                image_tensor = functional.to_tensor(img_query)	
                noisy_img_query = functional.adjust_saturation(image_tensor, saturation_factor=saturation_factor)	
            #    print('saturate', type(noisy_img_query))	
            else:
                noisy_img_query = img_query

                #raise ValueError(f"Unsupported noise type: {noise_type}")
            # self.noisy_images[self.id_list[idx][1]] = noisy_img_query
            
            if isinstance(noisy_img_query, torch.Tensor):
                noisy_img_query = functional.to_pil_image(noisy_img_query)
            elif isinstance(noisy_img_query, Image.Image):
                # No need for conversion, already a PIL Image
                pass

            # Apply transformation to preprocess the image
            noisy_img_query = self.transform_query(noisy_img_query)
            
            img_query = self.transform_query(img_query)
            
            return [img_query, noisy_img_query], img_reference, torch.tensor(idx), (idx), 0, 0

        elif 'scan_val' in self.mode:
            img_reference = Image.open(self.root + self.id_test_list2[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            img_query = Image.open(self.root + self.id_test_list2[index][1]).convert('RGB')
         #   img_query = shot_noise(img_query)
         #   img_query = Image.fromarray(np.uint8(img_query))  # Convert NumPy array back to PIL Image
            img_query = self.transform_query(img_query)
            return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0

        elif 'test_reference' in self.mode:
         #  img_reference = Image.open('/home/ak362297/TransGeo2022/CVUSANoise_CSVs/' + self.id_test_list[index][0]).convert('RGB')
            img_reference = Image.open(self.root + self.id_test_list2[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val',str(index)+'.png')).convert('RGB')
                return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
            return img_reference, torch.tensor(index), 0

        elif 'test_query' in self.mode:
            idx = index % len(self.id_idx_list)
            #img_query = Image.open('/home/c3-0/parthpk/CVUSA/' + self.id_list[idx][1]).convert('RGB')
            img_query = Image.open(self.root + self.id_test_list2[index][1]).convert('RGB')
            noise_type = 'Gaussian Noise'
        
            if isinstance(img_query, torch.Tensor):
                img_query = to_pil_image(img_query)

            if noise_type == 'Gaussian Noise':	
                noisy_img_query = add_gaussian_noise(img_query, mean=0.0, std=self.severity)	
            if noise_type == 'Brightness':	
                brightness_factors = [1.2, 1.4, 1.6, 1.8, 2.0]
                enhancer = ImageEnhance.Brightness(img_query)	
                noisy_img_query = enhancer.enhance(brightness_factors[self.severity - 1])
         #      print('brightness', type(noisy_img_query))
            elif noise_type == 'Contrast':	
                contrast_factors = [0.4, 0.3, 0.2, 0.1, 0.05]
                contrast_factor = contrast_factors[self.severity - 1]
                noisy_img_query = functional.adjust_contrast(img_query, contrast_factor=contrast_factor)
           #    print('Contrast', type(noisy_img_query))
            elif noise_type == 'Gaussian Blur':	
                radius = [1, 2, 3, 4, 5][self.severity - 1]	
                noisy_img_query = img_query.filter(ImageFilter.GaussianBlur(radius))
           #    print('gaussian blur', type(noisy_img_query))
            elif noise_type == 'Hue':
                img_hue_factor  = [0.04, 0.08, 0.12, 0.16, 0.2][self.severity - 1]
                noisy_img_query = functional.adjust_hue(img_query, img_hue_factor)
           #    print('hue', type(noisy_img_query))	
            elif noise_type == 'Saturate':	
                saturation_factor = [0.5, 0.75, 1.0, 1.25, 1.5][self.severity - 1]	
                noisy_img_query = functional.adjust_saturation(img_query, saturation_factor=saturation_factor)	
            #    print('saturate', type(noisy_img_query))	
            else:
                noisy_img_query = img_query

            if isinstance(noisy_img_query, torch.Tensor):
                noisy_img_query = functional.to_pil_image(noisy_img_query)
            elif isinstance(noisy_img_query, Image.Image):
                # No need for conversion, already a PIL Image
                pass

            noisy_img_query = self.transform_query(noisy_img_query)
     #      img_query_pil = to_pil_image(img_query)  # Convert tensor back to PIL Image
     #      img_query_pil.save("/home/ak362297/TransGeo2022/output.jpg")
            return noisy_img_query, torch.tensor(index), torch.tensor(index)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'scan_val' in self.mode:
            return len(self.id_test_list2)
        elif 'test_reference' in self.mode:
            return len(self.id_test_list2)
        elif 'test_query' in self.mode:
            return len(self.id_test_list2)
        else:
            print('not implemented!')
            raise Exception
