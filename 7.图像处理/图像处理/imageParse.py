from __future__ import print_function
from PIL import Image
import os, sys
img = Image.open('../images/horse-1.jpg')
print(img.format, img.mode, img.size)

# 创建缩略图, 缩略图的储存不超过给出的size
def thumbnailImage():
    size = [128, 128]
    box = (10, 100, 200, 200)
    for root, dirs, files in os.walk('../images'):
        print(files)
        for infile in files:
            outfile = os.path.join(root, infile.split(".")[0] + "-thumbnail.jpg")
            if infile != outfile:
                try:
                    im = Image.open(os.path.join(root, infile))
                    region = im.crop(box)
                    region = region.transpose(Image.ROTATE_180)
                    im.paste(region, box)
                    im.show()
                    im.thumbnail(size)
                    im.save(outfile, "JPEG")
                    print(outfile, im.size)
                except IOError:
                    print("Parse output thumbnail image fail")

def rollImage(image, delta):
    x_size, y_size = image.size
    delta = delta % x_size
    if delta == 0:
        return image
    part1 = image.crop((0, 0, delta, y_size))
    part2 = image.crop((delta, 0, x_size, y_size))
    image.paste(part1, (0, 0, x_size - delta, y_size))
    image.paste(part2, (x_size - delta, 0, x_size, y_size))
    return image

def channelSplit(img):
    r, g, b = img.split()
    return b
    # img = Image.merge("")
# rollImage(img, 2).show()


def resizeImage(img):
    return img.resize((100, 100))

def rotateImage(img):
    return img.rotate(-45)

def transposeImage(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

# channelSplit(img).show()
# resizeImage(img).show()
# rotateImage(img).show()
# transposeImage(img).show()