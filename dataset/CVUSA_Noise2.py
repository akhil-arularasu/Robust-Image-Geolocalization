import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
import sys
import cv2
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import random
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


# pytorch version of CVUSA loader
class CVUSA_Noise2(torch.utils.data.Dataset):

    def __init__(self, mode = '', root = '/home/c3-0/parthpk/CVUSA/', query_root = '/home/ak362297/TransGeo2022/FINALCVUSANoiseSeverity2/',same_area=True, print_bool=False, polar = '', args=None, noiseName='Gaussian Noise'): #CV-dataset
        super(CVUSA_Noise2, self).__init__()
        self.noiseName = noiseName
        self.args = args
        self.root = root
        self.query_root = query_root
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
        self.test_list = '/home/ak362297/TransGeo2022/CVUSANoise_CSVs/' + noiseName + '.csv' # desired directory for csv file
        self.test_listReference = self.root + 'splits/val-19zl.csv'

        test_refListlen = 0
 #      self.id_reftest_list = []
        self.id_reftest_idx_list = []
        with open(self.test_listReference, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                pano_id = (data[0].split('/')[-1]).split('.')[0]
             #   print('pano_id' + pano_id)
             #   satellite filename, streetview filename, pano_id
             #   self.id_reftest_list.append([data[0], data[1], pano_id])
                self.id_reftest_idx_list.append(idx)
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
        
        self.noisy_images = {}  # Dictionary to store noisy images
        
        if self.mode == 'train':
            print('generating noisy images for the entire training dataset')
            # Generate and store noisy images for each training image
            for i in range(self.data_size):
                img_query_path = '/home/c3-0/parthpk/CVUSA/' + self.id_list[i][1]
                img_query = Image.open(img_query_path).convert('RGB')
                # Apply random noise to img_query and store the noisy image
                noisy_img_query = self.apply_random_noise(img_query)
                print(self.id_list[i][1])
                self.noisy_images[self.id_list[i][1]] = noisy_img_query

    def __getitem__(self, index, debug=False):
        if self.mode == 'train':
            idx = index % len(self.id_idx_list)
            img_query = Image.open('/home/c3-0/parthpk/CVUSA/' + self.id_list[idx][1])
            print('going in getitem')
            # Convert image to RGB if it's not already in RGB format
            if img_query.mode != 'RGB':
                img_query = img_query.convert('RGB')
            img_reference = Image.open('/home/c3-0/parthpk/CVUSA/' + self.id_list[idx][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)

            noisy_img_query = self.noisy_images[self.id_list[idx][1]]

            noisy_img_query = np.uint8(noisy_img_query)
            noisy_img_query = np.reshape(noisy_img_query, (noisy_img_query.shape[0], noisy_img_query.shape[1], 3))

            # Convert NumPy array to PIL Image
            noisy_img_query = Image.fromarray(noisy_img_query)

            # Apply transformation to preprocess the image
            noisy_img_query = self.transform_query(noisy_img_query)
                            
            img_query = self.transform_query(img_query)
            
            return [img_query, noisy_img_query], img_reference, torch.tensor(idx), (idx), 0, 0

        elif 'scan_val' in self.mode:
            img_reference = Image.open(self.query_root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            img_query = Image.open(self.query_root + self.id_test_list[index][1]).convert('RGB')
         #   img_query = shot_noise(img_query)
         #   img_query = Image.fromarray(np.uint8(img_query))  # Convert NumPy array back to PIL Image
            img_query = self.transform_query(img_query)
            return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0

        elif 'test_reference' in self.mode:
         #  img_reference = Image.open('/home/ak362297/TransGeo2022/CVUSANoise_CSVs/' + self.id_test_list[index][0]).convert('RGB')
            img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
            img_reference = self.transform_reference(img_reference)
            if self.args.crop:# and index <= 8883:
                atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val',str(index)+'.png')).convert('RGB')
                return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
            return img_reference, torch.tensor(index), 0

        elif 'test_query' in self.mode:
            idx = index % len(self.id_idx_list)
            img_query = Image.open(self.query_root + self.id_test_list[index][1]).convert('RGB')
        #   print('img_query')
        #   print(self.query_root + self.id_test_list[index][1])
            img_query = self.transform_query(img_query)
     #      img_query_pil = to_pil_image(img_query)  # Convert tensor back to PIL Image
     #      img_query_pil.save("/home/ak362297/TransGeo2022/output.jpg")
            return img_query, torch.tensor(index), torch.tensor(index)
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'scan_val' in self.mode:
            return len(self.id_test_list)
        elif 'test_reference' in self.mode:
            print('lennn::')
            print(len(self.id_reftest_idx_list))
            return len(self.id_reftest_idx_list)
        elif 'test_query' in self.mode:
            return len(self.id_test_list)
        else:
            print('not implemented!')
            raise Exception
    
    def apply_random_noise(self, img_query):
        # Apply random noise to img and return the noisy image
        noise_type = random.choice(['Gaussian Noise', 'Shot Noise', 'Impulse Noise', 'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur', 'Snow', 'Fog', 'Brightness', 'Contrast', 'Elastic', 'Pixelate', 'JPEG', 'Speckle Noise', 'Gaussian Blur', 'Spatter', 'Saturate'])      
        severity = 1  # Assuming you have a list of severity levels
        if noise_type == 'Gaussian Noise':
            noisy_img_query = trainingNoiseCluster.gaussian_noise(np.array(img_query), severity)
        elif noise_type == 'Shot Noise':
            noisy_img_query = trainingNoiseCluster.shot_noise(img_query, severity)
        elif noise_type == 'Impulse Noise':
            noisy_img_query = trainingNoiseCluster.impulse_noise(img_query, severity)
        elif noise_type == 'Defocus Blur':
            noisy_img_query = trainingNoiseCluster.defocus_blur(img_query, severity)
        elif noise_type == 'Glass Blur':
            noisy_img_query = trainingNoiseCluster.glass_blur(img_query, severity)
        elif noise_type == 'Motion Blur':
            noisy_img_query = trainingNoiseCluster.motion_blur(img_query, severity)
        elif noise_type == 'Zoom Blur':
            noisy_img_query = trainingNoiseCluster.zoom_blur(img_query, severity)
        elif noise_type == 'Snow':
            noisy_img_query = trainingNoiseCluster.snow(img_query, severity)
        elif noise_type == 'Fog':
            noisy_img_query = trainingNoiseCluster.fog(img_query, severity)
        elif noise_type == 'Brightness':
            noisy_img_query = trainingNoiseCluster.brightness(img_query, severity)
        elif noise_type == 'Contrast':
            noisy_img_query = trainingNoiseCluster.contrast(img_query, severity)
        elif noise_type == 'Elastic':
            noisy_img_query = trainingNoiseCluster.elastic_transform(img_query, severity)
        elif noise_type == 'Pixelate':
            noisy_img_query = trainingNoiseCluster.pixelate(img_query, severity)
        elif noise_type == 'JPEG':
            noisy_img_query = trainingNoiseCluster.jpeg_compression(img_query, severity)
        elif noise_type == 'Speckle Noise':
            noisy_img_query = trainingNoiseCluster.speckle_noise(img_query, severity)
        elif noise_type == 'Gaussian Blur':
            noisy_img_query = trainingNoiseCluster.gaussian_blur(np.array(img_query), severity)
        elif noise_type == 'Spatter':
            noisy_img_query = trainingNoiseCluster.spatter(img_query, severity)
        elif noise_type == 'Saturate':
            noisy_img_query = trainingNoiseCluster.saturate(img_query, severity)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        return noisy_img_query

'''     
        ## UPDATING CSV FILE
        with open(self.test_list, 'r') as file:
            lines = file.readlines()
        modified_lines = []
        for line in lines:
            data = line.strip().split(',')
            if len(data) >= 2:
                data[1] = noiseName + '/' + data[1].split('/')[-1]
            modified_line = ','.join(data)
            modified_lines.append(modified_line)
'''    

'''            
        with open(self.test_list, 'w') as file:
            file.write('\n'.join(modified_lines))
'''