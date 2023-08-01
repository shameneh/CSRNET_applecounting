import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''
    def __init__(self,phase='train'):
        self.dataset_root = '/home/ubuntu/ameneh/dataset/all_data'
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr           = 1e-4                # learning rate
        self.batch_size   = 8                   # batch size
        self.epochs       = 2000                # epochs
        self.checkpoints  = './checkpoints_weighted_loss'     # checkpoints dir 
        self.writer       = SummaryWriter()     # tensorboard writer
        self.image_size   = (1280,640) 
        self.lds          = True 
        self.__mkdir(self.checkpoints,phase)

    def __mkdir(self, path,phase):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path) and phase == 'train':
            os.makedirs(path)
            print('create dir: ',path)
