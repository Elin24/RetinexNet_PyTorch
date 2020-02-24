import numpy as np
from PIL import Image
import torchvision.transforms as TTF
import random
import os

def randomCrop(imgs, size):
    if type(size) == int:
        size = (size, size)
    w, h = imgs[0].size
    ws, hs = random.randint(0, w - size[0]), random.randint(0, h - size[1])
    wt, ht = ws + size[0], hs + size[1]
    return [img.crop((ws, hs, wt, ht)) for img in imgs]


img2tensor = TTF.ToTensor()
train_data_augmentation = TTF.Compose([
    TTF.Lambda(lambda imgs: imgs if random.random() > 0.5 else [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]),
    TTF.Lambda(lambda imgs: imgs if random.random() > 0.5 else [img.transpose(Image.FLIP_TOP_BOTTOM) for img in imgs]),
    TTF.Lambda(lambda imgs: [img2tensor(img) for img in imgs]),
    TTF.Lambda(lambda xs: xs if random.random() > 0.5 else [x.permute(0, 2, 1) for x in xs])
])

eval_data_augmentation = TTF.Compose([
    TTF.Lambda(lambda imgs: [img2tensor(img) for img in imgs])
])
test_data_augmentation = TTF.Compose([
    TTF.ToTensor()
])

def saveimg(img, savedir, imgname):
    os.makedirs(savedir, exist_ok=True)
    img.save(os.path.join(savedir, imgname))