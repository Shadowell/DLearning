import d2lzh as d2l
from mxnet import image
import gluonbook as gb
from mxnet.gluon import data as gdata

d2l.set_figsize()
img = image.imread('../images/cat.jpg').asnumpy()
d2l.plt.imshow(img)

cat_box = [60, 20, 520, 480]

def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(xy=bbox[0], bbox[1], width=bbox[2] - bbox[0], height=bbox[3] - bbox[1]
                             , fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(image)
fig.axes.add_pat

