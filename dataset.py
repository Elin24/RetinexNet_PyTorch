import random
from PIL import Image
import os
import torch.utils.data as data
from glob import glob
import utils
import torchvision.transforms as TTF

# folder structure
# data
#   |train
#   |  |low
#   |  |  |*.png
#   |  |high
#   |  |  |*.png
#   |eval
#   |  |low
#   |  |  |*.png
def get_dataset_len(route, phase):
    if phase in ['train', 'valid']:
        low_data_names = glob(route + phase + '/low/*.png')
        high_data_names = glob(route + phase + '/high/*.png')
        low_data_names.sort()
        high_data_names.sort()
        assert len(low_data_names) == len(high_data_names)
        return len(low_data_names), [low_data_names, high_data_names]
    elif phase == 'test':
        low_data_names = glob(route + '/*.png')
        return len(low_data_names), low_data_names
    else:
        return 0, []

class TheDataset(data.Dataset):
    def __init__(self, route='./data/', phase='train', patch_size=400):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size
        self.len, self.data_names = get_dataset_len(route, phase)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.phase == 'test':
            low_im = Image.open(self.data_names[item])
            return utils.test_data_augmentation(low_im), os.path.basename(self.data_names[item])
        
        low_im = Image.open(self.data_names[0][item])
        high_im = Image.open(self.data_names[1][item])
        if self.phase == 'train':
            low_im, high_im = utils.randomCrop([low_im, high_im], self.patch_size)
            low_im, high_im = utils.train_data_augmentation([low_im, high_im])
        elif self.phase == 'eval':
            low_im, high_im = utils.test_data_augmentation([low_im, high_im])
        return low_im, high_im
