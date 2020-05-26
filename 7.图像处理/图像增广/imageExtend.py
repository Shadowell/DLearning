import sys
sys.path.insert(0, '..')

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import matplotlib as plt
from time import time
from PIL import Image


gb.set_figsize()
img = image.imread('../images/cat.jpg')
#img2 = Image.open('../images/dog.jpg')
#img2.show()
gb.plt.imshow(img.asnumpy())

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = gb.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

# 左右翻转
apply(img, gdata.vision.transforms.RandomFlipLeftRight())

# 上下翻转
#apply(img, gdata.vision.transforms.RandomFlipTopBottom)

# 每次随机裁剪⼀⽚⾯积为原⾯积 10% 到 100% 的区域，其宽和⾼的⽐例在 0.5 和 2 之间，然后再将⾼宽缩放到 200 像素⼤小
shape_aug = gdata.vision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 随机亮度改为原图的 50% 到 150%
apply(img, gdata.vision.transforms.RandomBrightness(0.5))

# 可以修改⾊相
apply(img, gdata.vision.transforms.RandomHue(0.5))

# 使⽤ RandomColorJitter 来⼀起使⽤
color_aug = gdata.vision.transforms.RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 联合多个增广变换
augs = gdata.vision.transforms.Compose[gdata.vision.transforms.RandomFlipTopBottom(), color_aug, shape_aug]
apply(img, augs)