# Aditya Gupta, January 23
# extract predictions using nms

import numpy as np
import collections
import cv2

eps = 0.00001
d_th = 0.5
nms_th = 0.5
prediction_list = ['/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/285.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/757.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/1955.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2149.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2164.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2759.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/4229.npy'
                    ]

img_list = ['/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000285.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000757.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000001955.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002149.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002164.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002759.jpg',
            '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000004229.jpg'
            ]

final_list = []
model_img_dim = [416, 416] #[width height]

def iou(boxA, boxB):
    # claculate iou between two boxes
    x_a = max(boxA[0], boxB[0])
    x_b = min(boxA[2], boxB[2])
    y_a = max(boxA[1], boxB[1])
    y_b = min(boxA[3], boxB[3])

    anb = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    result = anb / float(area_B + area_A - anb + eps)

    return result


def xywh2xyxy(box):
    # convert to [centre-xy,h,w] to [top-left xy, bottom-right xy]
    bbox = [0]*4
    bbox[0] = box[0] - (box[2] / 2)
    bbox[1] = box[1] - (box[3] / 2)
    bbox[2] = box[0] + (box[2] / 2)
    bbox[3] = box[1] + (box[3] / 2)
    return bbox


def resize_to_orignal_img(bbox, img_dim, model_dim):
    # resize to orignal size of image
    box = [0]*4
    box[0] = (bbox[0] * img_dim[0]) / model_dim[0]
    box[1] = (bbox[1] * img_dim[1]) / model_dim[1]
    box[2] = (bbox[2] * img_dim[0]) / model_dim[0]
    box[3] = (bbox[3] * img_dim[1]) / model_dim[1]
    return box


def apply_nms(pred, thresh, img_dim, model_dim):
    cls_pred = collections.defaultdict(list)
    final_cls_pred = collections.defaultdict(list)
    # final_cls_pred_orig = collections.defaultdict(list)
    # sort precitions based on classes
    for p in pred:
        cls_pred[p[1]].append(p[0])

    for k in cls_pred.keys():
        # for each class, sort on objectness score
        cls_pred[k] = sorted(cls_pred[k], key=lambda x: x[4])
        cls_pred[k].reverse()
        while len(cls_pred[k]):
            # comapre with highest score and remove if exceeds threshold
            current_max_pred = cls_pred[k][0]
            if len(cls_pred[k]) == 1:
                final_cls_pred[k].append(cls_pred[k][0])
                # final_cls_pred_orig[k].append(cls_pred[k][0])
                cls_pred[k].pop()
            else:
                to_remove = []
                for index in range(1, len(cls_pred[k])):
                    current_iou = iou(xywh2xyxy(current_max_pred[:-1]),
                                      xywh2xyxy(cls_pred[k][index][:-1]))
                    if current_iou > nms_th:
                        to_remove.append(index)
                cls_pred[k] = [i for j, i in enumerate(cls_pred[k]) if j not in to_remove]
                # convert to image size and top-left, bottom-right coordinate system
                final_cls_pred[k].append(resize_to_orignal_img(xywh2xyxy(cls_pred[k][0][:-1]),img_dim, model_dim))
                cls_pred[k].pop(0)

    # # print
    # for j in final_cls_pred.keys():
    #     print('key : ', j)
    #     for k, l in enumerate(final_cls_pred[j]):
    #         print(k, l)

    return final_cls_pred


def run():
    for ii,i in enumerate(prediction_list):
        print('+-----+-----+-----+')
        print(i.split('/')[-1])
        current_predictions = np.load(i)[0]
        valid_pred = []

        # 1) remove predictions with less confidence
        for j in current_predictions:
            if j[4] > d_th:
                valid_pred.append([j[0:5], np.argmax(j[5:]), j[5 + np.argmax(j[5:])]])
        # 2) apply nms
        img_dim = cv2.imread(img_list[ii]).shape #[height, width, channel
        img_dim = [img_dim[1],img_dim[0]]        #[width, height]
        final_list.append(apply_nms(valid_pred, nms_th, img_dim, model_img_dim))


if __name__ == "__main__":
    run()

print('Done!')
