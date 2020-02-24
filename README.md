# Retinex_PyTorch

This is a unofficial Pytorch implementation of RetinexNet for low light enhancement.

## paper

[Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/pdf/1808.04560), in  In BMVC 2018.

## Results

![22 and 547](sample/retinex.png)

## train & test

### train
```
# illumination channel number = 1
python train.py --Ichannel 1 --checkpoint_dir ./checkpoint_1 --ngpus 2
# illumination channel number = 3
python train.py --Ichannel 3 --checkpoint_dir ./checkpoint_3 --ngpus 2
```
###  test
```
# illumination channel number = 1
python test.py --Ichannel 1 --decomnet_path ./checkpoint_1/decom_final.pth --relightnet_path ./checkpoint_1/relight_final.pth --save_path outimg_1
# illumination channel number = 3
python test.py --Ichannel 3 --decomnet_path ./checkpoint_3/decom_final.pth --relightnet_path ./checkpoint_3/relight_final.pth --save_path outimg_3
```

## Citation

```
@inproceedings{zhang2019kindling,
 author = {Zhang, Yonghua and Zhang, Jiawan and Guo, Xiaojie},
 title = {Kindling the Darkness: A Practical Low-light Image Enhancer},
 booktitle = {Proceedings of the 27th ACM International Conference on Multimedia},
 series = {MM '19},
 year = {2019},
 isbn = {978-1-4503-6889-6},
 location = {Nice, France},
 pages = {1632--1640},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3343031.3350926},
 doi = {10.1145/3343031.3350926},
 acmid = {3350926},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {image decomposition, image restoration, low light enhancement},
}
```