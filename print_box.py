import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# img = mpimg.imread(
    # '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000001955.jpg')
# img = Image.open('/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000001955.jpg')    # Open image as PIL image object
# img = Image.open('/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000757.jpg')    # Open image as PIL image object
img = Image.open('/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000004229.jpg')    # Open image as PIL image object
# img = img.resize((np.array([416,416])).astype(int))
imgplot = plt.imshow(img)


fig, ax = plt.subplots()
ax.imshow(img)
######### ######### ######### ######### ######### ######### ######### #########
######### ######### #########                     ######### ######### #########
######### ######### ######### ######### ######### ######### ######### #########
# 1955 :: correct answer (left-xy, width, height)
# [144.49033187379808, 458.84695384615384, 65.12442130048076, 179.7641230769231]
# [172.59460606971152, 120.37947692307694, 167.9413273798077, 214.39513846153847]
######### ######### ######### ######### ######### ######### ######### #########
# 1955 :: correct answer (left-left-xy, bottom-right-xy)
# [144.4903322137319, 458.84695199819714, 209.6147535259907, 638.611074594351]
# [172.59459974215582, 120.37947434645433, 340.5359322291154, 334.77461594801684]
######### ######### ######### ######### ######### ######### ######### #########
######### ######### #########                     ######### ######### #########
######### ######### ######### ######### ######### ######### ######### #########

# bb = [[400.24537306565503, 267.54009532928467, 635.2107943021334, 338.43328923445483],
#       [4.835533728966346, 146.84231207920953, 348.3421443058894, 335.7003740897545],
#       [259.14759709284857, 74.80461425047655, 617.2305767352765, 224.8113264303941],
#       [210.94037,   210.5232,    381.18063,   261.57108],
#       [135.32690048217773, 282.02761895381485, 178.39105606079102, 296.7964959190442],
#       [23.853794978215145, 252.1683627137771, 81.06647198016827, 284.0342763524789]]
# colors = ['b', 'r','g','b', 'r','g']

# 1955
# bb = [[144.4903322137319, 458.84695199819714, 209.6147535259907, 638.611074594351],
#       [172.59459974215582, 120.37947434645433, 340.5359322291154, 334.77461594801684]]
# colors = ['b','r']
# bb =[[172.49147,  356.67386,   63.446743, 116.84668],
#      [249.95586, 147.92508, 163.61497, 139.35684]]

# 757
# bb = [[75.91327373798077, 89.55381793242235, 574.5356633112981, 347.706564279703],
#       [346.88754,    200.16357,    142.91997,    277.75577]]
# colors = ['b','r']

# 4229
bb= [[131.7887,  220.16098, 169.201,   140.31512],
     [229.37048, 198.51242,  99.57041, 119.35078],
     [308.6762,  244.21274, 214.88812, 102.89835],
     [405.90646,  161.40353,   20.761972,  35.093666]]
colors = ['b','r','m','g']

def xyc_to_xytl(bbox):
    # convert centre xy to top-left xy, width and height in both
    box = [0]*len(bbox)
    box[0] = bbox[0] - (bbox[2] / 2)
    box[1] = bbox[1] - (bbox[3] / 2)
    box[2] = bbox[2]# + (bbox[2] / 2)
    box[3] = bbox[3]# + (bbox[3] / 2)
    return box

def resize_to_orignal_img(bbox):
    #this way is better than appending
    # TODO : add checks for side cases
    # img_dim   = [427, 640] #1955
    img_dim   = [640, 427] #1955
    model_dim = [416, 416]

    box = [0]*4
    box[0] = (bbox[0] * img_dim[0]) / model_dim[0]
    box[1] = (bbox[1] * img_dim[1]) / model_dim[1]
    box[2] = (bbox[2] * img_dim[0]) / model_dim[0]
    box[3] = (bbox[3] * img_dim[1]) / model_dim[1]
    return box


for i, a in enumerate(bb):
    [x, y, w, h] = resize_to_orignal_img(xyc_to_xytl(a))
    print([x,y,w,h])
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=colors[i], facecolor='none')
    plt.plot(x, y, marker='o', color="white")
    ax.add_patch(rect)

plt.show()