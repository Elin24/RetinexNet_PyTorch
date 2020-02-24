import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from model import DecomNet, RelightNet
from dataset import TheDataset
from loss import DecomLoss, RelightLoss
import tqdm
import utils

parser = argparse.ArgumentParser(description='RetinexNet args setting')
parser.add_argument('--Ichannel', dest='Ichannel', type=int, default=1, help='Illumination channels, 1 or 3')
parser.add_argument('--ngpus', dest='ngpus', type=int, default=4, help='number of GPUs')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=400, help='patch size')
parser.add_argument('--workers', dest='workers', type=int, default=16, help='num workers of dataloader')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--save_interval', dest='save_interval', type=int, default=20, help='save model every # epoch')
parser.add_argument('--datapath', dest='datapath', default='./data/', help='path for loading test images')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint_1', help='directory for checkpoints')
parser.add_argument('--decom', dest='decom', default=0,
                    help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

decom_net = DecomNet(args.Ichannel)
relight_net = RelightNet(args.Ichannel)

if args.ngpus > 0:
    gpus = [_ for _ in range(args.ngpus)]
    decom_net = torch.nn.DataParallel(decom_net, device_ids=gpus).cuda()#decom_net.cuda()
    relight_net = torch.nn.DataParallel(relight_net, device_ids=gpus).cuda()#relight_net.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

lr = args.start_lr * np.ones([args.epoch])
lr[20:] = lr[0] / 10.0

decom_optim = torch.optim.Adam(decom_net.parameters(), lr=args.start_lr)
relight_optim = torch.optim.Adam(relight_net.parameters(), lr=args.start_lr)

train_set = TheDataset(phase='train', patch_size=args.patch_size)

decom_criterion = DecomLoss(args.Ichannel)
relight_criterion = RelightLoss(args.Ichannel)

def train():
    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
        decom_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()

            decom_optim.zero_grad()
            _, r_low, l_low = decom_net(low_im)
            _, r_high, l_high = decom_net(high_im)
            loss = decom_criterion(r_low, l_low, r_high, l_high, low_im, high_im)
            loss.backward()
            decom_optim.step()

            sum_loss += loss

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch+1) % args.save_interval == 0:
            torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_' + str(epoch) + '.pth')

    torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_final.pth')

    for para in decom_net.parameters():
        para.requires_grad = False

    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
        relight_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()

            relight_optim.zero_grad()
            lr_low, r_low, _ = decom_net(low_im)
            l_delta = relight_net(lr_low.detach())
            loss = relight_criterion(l_delta, r_low.detach(), high_im)
            loss.backward()
            relight_optim.step()

            sum_loss += loss

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch+1) % args.save_interval == 0:
            torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_' + str(epoch) + '.pth')

    torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_final.pth')



if __name__ == '__main__':
    train()
