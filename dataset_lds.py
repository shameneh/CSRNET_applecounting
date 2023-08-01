#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
class CrowdDataset(torch.utils.data.Dataset):
    '''
    CrowdDataset
    '''

    def __init__(self, root, phase,image_size=None, main_transform=None, img_transform=None, dmap_transform=None, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        '''
        root: the root path of dataset.
        phase: train or test.
        main_transform: transforms on both image and density map.
        img_transform: transforms on image.
        dmap_transform: transforms on densitymap.
        '''
        self.image_size = image_size
        self.img_path = os.path.join(root, phase+'/images')
        self.dmap_path = os.path.join(root, phase+'/densitymaps')
        self.data_files = [filename for filename in os.listdir(self.img_path)
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.dmap_transform = dmap_transform
        self.weights = self._prepare_weights(lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        index = index % len(self.data_files)
        fname = self.data_files[index]
        img, dmap = self.read_image_and_dmap(fname)
        if self.main_transform is not None:
            img, dmap = self.main_transform((img, dmap))
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dmap_transform is not None:
            dmap = self.dmap_transform(dmap)
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        return {'image': img, 'densitymap': dmap, 'image_name':fname,'weight': weight}

    def read_image_and_dmap(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            print('There is a grayscale image.')
            img = img.convert('RGB')
        if self.image_size is not None:
            img = img.resize(self.image_size)
        dmap = np.load(os.path.join(
            self.dmap_path, os.path.splitext(fname)[0] + '.npy'))
        dmap = dmap.astype(np.float32, copy=False)
        dmap = Image.fromarray(dmap)
        return img, dmap
    def _prepare_weights(self,  max_target=100, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        print('weights preparing ...')
        if lds:
            value_dict = {x: 0 for x in range(max_target)}
            labels =[]
            for fname in self.data_files:
                dmap = np.load(os.path.join(self.dmap_path, os.path.splitext(fname)[0] + '.npy'))
                dmap = dmap.astype(np.float32, copy=False)
                dmap_img = Image.fromarray(dmap)
                labels.append(dmap.sum())
            # mbr
            print(len(labels),len(self.data_files))
            for label in labels:
                value_dict[min(max_target - 1, int(label))] += 1
            num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
            if not len(num_per_label):
                print(f"there is no lable")
                return None

        #if lds:
            lds_kernel_window = self.get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
            #print(value_dict)
            
            weights = [np.float32(1 / x) for x in num_per_label]

            scaling = len(weights) / np.sum(weights)
            #print(labels[0],weights[0],scaling,scaling*weights[0])
            
            weights = [scaling * x for x in weights]

            #another method
            #beta = 0.9
            #effective_num = 1.0 - np.power(beta, num_per_label)
            #weights_1 = (1.0 - beta)/ np.array(effective_num)
            #weights_1 = weights_1 / np.sum(weights)*max_target
            
           # print(weights[0],weights_1[0],labels[0])
          
            return weights
        else:
            return None

    def get_lds_kernel_window(self,kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window


def create_train_dataloader(root, use_flip,image_size, batch_size,lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    '''
    Create train dataloader.
    root: the dataset root.
    use_flip: True or false.
    batch size: the batch size.
    '''
    main_trans_list = []
    if use_flip:
        main_trans_list.append(RandomHorizontalFlip())
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor()])#, Normalize(mean=[0.5,0.5,0.5],std=[0.225,0.225,0.225])])
    dmap_trans = ToTensor()
    dataset = CrowdDataset(root=root, phase='train',image_size =image_size, main_transform=main_trans, 
                    img_transform=img_trans,dmap_transform=dmap_trans,lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers = 8)
    return dataloader

def create_test_dataloader(root,image_size):
    '''
    Create train dataloader.
    root: the dataset root.
    '''
    main_trans_list = []
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor()])#, Normalize(mean=[0.5,0.5,0.5],std=[0.225,0.225,0.225])])
    dmap_trans = ToTensor()
    dataset = CrowdDataset(root=root, phase='validation',image_size=image_size, main_transform=main_trans, 
                    img_transform=img_trans,dmap_transform=dmap_trans)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, num_workers =8)
    return dataloader

#----------------------------------#
#          Transform code          #
#----------------------------------#
class RandomHorizontalFlip(object):
    '''
    Random horizontal flip.
    prob = 0.5
    '''
    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap
        if random.random() < 0.5:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            return (img, dmap)

class PairedCrop(object):
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network, 
    we must promise that the size of input image is the corresponding factor.
    '''
    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap
        
        i, j, th, tw = self.get_params(img, self.factor)

        img = F.crop(img, i, j, th, tw)
        dmap = F.crop(dmap, i, j, th, tw)
        return (img, dmap)


# testing code
# if __name__ == "__main__":
#     root = './data/part_B_final'
#     dataloader = create_train_dataloader(root, True, 2)
#     for i, data in enumerate(dataloader):
#         image = data['image']
#         densitymap = data['densitymap']
#         print(image.shape,densitymap.shape)
