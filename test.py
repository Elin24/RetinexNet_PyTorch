import os
import torch
import torchvision.transforms as TTF
import numpy as np
import argparse
from model import DecomNet, RelightNet
from dataset import TheDataset
from loss import DecomLoss, RelightLoss
import tqdm
import utils

parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--Ichannel', dest='Ichannel', type=int, default=1, help='Illumination channels, 1 or 3')
parser.add_argument('--ngpus', dest='ngpus', default='1', type=int, help='number of GPUs')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number of samples in one batch')
parser.add_argument('--workers', dest='workers', type=int, default=1, help='num workers of dataloader')
parser.add_argument('--datapath', dest='datapath', default='data/eval/low', help='path for loading test images')
parser.add_argument('--decomnet_path', dest='decomnet_path', default='./checkpoint_1/decom_final.pth', help='path for loading decomnet')
parser.add_argument('--relightnet_path', dest='relightnet_path', default='./checkpoint_1/relight_final.pth', help='path for loading relightnet')
parser.add_argument('--save_path', dest='save_path', default='outimg_1', help='path for save output enlight image')

args = parser.parse_args()

decom_net = DecomNet(args.Ichannel)
decom_net.load_state_dict(torch.load(args.decomnet_path))
relight_net = RelightNet(args.Ichannel)
relight_net.load_state_dict(torch.load(args.relightnet_path))

if args.ngpus > 0:
    gpus = [_ for _ in range(args.ngpus)]
    decom_net = decom_net.cuda()
    relight_net = relight_net.cuda()

test_set = TheDataset(route=args.datapath, phase='test')


def test():
    decom_net.eval()
    relight_net.eval()
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    toImg = TTF.ToPILImage()
    for data in dataloader:#tqdm.tqdm(dataloader):
        low_im, imname = data[0].cuda(), data[1][0]
        lr_low, r_part, _ = decom_net(low_im)
        l_delta = relight_net(lr_low)
        ldelta = (r_part * l_delta).clamp(0.0, 1.0)[0]
        utils.saveimg(toImg(ldelta.cpu()), args.save_path, imname)




if __name__ == '__main__':
    test()
