# train
# illumination channel number = 1
# CUDA_VISIBLE_DEVICES=1,0 python train.py --Ichannel 1 --checkpoint_dir ./checkpoint_1 --ngpus 2
# illumination channel number = 3
# CUDA_VISIBLE_DEVICES=2,3 python train.py --Ichannel 3 --checkpoint_dir ./checkpoint_3 --ngpus 2

# test
# illumination channel number = 1
# python train.py --Ichannel 1 --decomnet_path ./checkpoint_1/decomnet_final.pth --relightnet_path ./checkpoint_1/relightnet_final.pth --save_path outimg_1
# illumination channel number = 3
# python train.py --Ichannel 3 --decomnet_path ./checkpoint_3/decomnet_final.pth --relightnet_path ./checkpoint_3/relightnet_final.pth --save_path outimg_3