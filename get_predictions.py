# Aditya Gupta, January 23
# extract predictions using nms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import collections
import copy
import cv2

eps = 0.00001

prediction_list = ['/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/285.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/757.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/831.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/836.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/1955.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2149.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2164.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/2759.npy',
                   '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/4229.npy'
                   ]
img_list = [
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000285.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000757.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000831.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000000836.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000001955.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002149.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002164.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000002759.jpg',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/images/COCO_val2014_000000004229.jpg'
]
lbl_list = [
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000000285.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000000757.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000000831.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000000836.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000001955.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000002149.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000002164.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000002759.txt',
    '/home/gabbar/Desktop/naukri_2022/code/yolo_postprocessing/extract_boxes/data/labels/COCO_val2014_000000004229.txt'
]


def iou(boxA, boxB):
    """
    Calculate intersection over union between two boxes
    :param boxA: box 1 coordinate in format [top-left x,y, bottom-right x,y]
    :param boxB: box 2 coordinate in format [top-left x,y, bottom-right x,y]
    :return:
    result : iou of two boxes
    """
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
    """
    Convert from [centre-xy,h,w] to [top-left xy, bottom-right xy]
    :param box: box in format [centre xy, h, w]
    :return: bbox: to [top-left x,y, bottom-right x,y]
    """

    bbox = [0] * 4
    bbox[0] = box[0] - (box[2] / 2)
    bbox[1] = box[1] - (box[3] / 2)
    bbox[2] = box[0] + (box[2] / 2)
    bbox[3] = box[1] + (box[3] / 2)
    return bbox


def nxywhtotlbr(box, img_dim):
    """
    Convert from normalised [centre-xy,h,w] to [top-left xy, bottom-right xy]
    :param box: normalised coordinate in [centre-xy, h, w]
    :param img_dim: image dimension to convert into pixel format
    :return: bbox : pixel format [top-left xy, bottom-right xy]
    """
    bbox = [0] * 4
    bbox[0] = int((box[0] - (box[2] / 2)) * img_dim[0])
    bbox[1] = int((box[1] - (box[3] / 2)) * img_dim[1])
    bbox[2] = int((box[0] + (box[2] / 2)) * img_dim[0])
    bbox[3] = int((box[1] + (box[3] / 2)) * img_dim[1])
    return bbox


def resize_to_orignal_img(bbox, img_dim, model_dim):
    """
    resize to image dimensions from model-input dimension
    """
    # resize to orignal size of image
    box = [0] * 4
    box[0] = (bbox[0] * img_dim[0]) / model_dim[0]
    box[1] = (bbox[1] * img_dim[1]) / model_dim[1]
    box[2] = (bbox[2] * img_dim[0]) / model_dim[0]
    box[3] = (bbox[3] * img_dim[1]) / model_dim[1]
    return box


def get_valid_box(pred, nms_th, img_dim, model_dim, img_id):
    """
    Remove boxes with iou greater than nms_th, happens for each class
    :param pred: contains list of prposals with valid objectness scoare
    :param nms_th: nms threshold, 0.5
    :param img_dim: image dimension, as we are getting proposals in resized version
    :param model_dim: image dimension which model resized image to
    :param img_id: name of images
    :return: class-wise valid proposals, in image dimensions
    """
    cls_pred = collections.defaultdict(list)
    final_cls_pred = collections.defaultdict(list)
    # arrange predicitions based on classes
    for p in pred:
        cls_pred[p[1]].append(p[0])

    # for each class, sort on objectness score
    for k in cls_pred.keys():
        cls_pred[k] = sorted(cls_pred[k], key=lambda x: x[4]) # sort based on objectness score
        cls_pred[k].reverse()
        while len(cls_pred[k]):
            # comapre with highest score and remove if exceeds threshold
            current_max_pred = cls_pred[k][0]
            # only single box, nothing to compare
            if len(cls_pred[k]) == 1:
                final_cls_pred[k].append(
                    [img_id, resize_to_orignal_img(xywh2xyxy(cls_pred[k][0][:-1]), img_dim, model_dim),
                     cls_pred[k][0][-1]])
                cls_pred[k].pop()
            else:
                to_remove = []
                for index in range(1, len(cls_pred[k])):
                    # if iou greater than threshold, remove it
                    current_iou = iou(xywh2xyxy(current_max_pred[:-1]),
                                      xywh2xyxy(cls_pred[k][index][:-1]))
                    if current_iou > nms_th:
                        to_remove.append(index)
                cls_pred[k] = [i for j, i in enumerate(cls_pred[k]) if j not in to_remove]
                # convert to image size and top-left, bottom-right coordinate system
                final_cls_pred[k].append(
                    [img_id, resize_to_orignal_img(xywh2xyxy(cls_pred[k][0][:-1]), img_dim, model_dim),
                     cls_pred[k][0][-1]])
                cls_pred[k].pop(0)

    # # print
    # for j in final_cls_pred.keys():
    #     for k, l in enumerate(final_cls_pred[j]):
    #         print(k, " | class : ", j, l)

    return final_cls_pred


def apply_nms(pred_list, nms_th=0.5, objectness_th=0.5, model_img_dim=[416,416]):
    """
    arg:
        pred_list     : output files from feature extractor
                        10647 number of proposal from YOLO-V3
        nms_th        : threshold for IOU for NMS, 0.5 by default
        objectness_th : objectness value threshold, to remove less confident proposals
        model_img_dim : size of image, at which model resizes image to [width, height]
    return:
        final_list : final prediction from model (post-nms)
    """
    final_list = collections.defaultdict(list)
    for ii, i in enumerate(pred_list):

        current_predictions = np.load(i)[0]
        valid_pred = []  # contains proposals greater than objectness score

        # 1) remove predictions with less confidence
        for j in current_predictions:
            if j[4] > objectness_th:
                valid_pred.append([j[0:5], np.argmax(j[5:]), j[5 + np.argmax(j[5:])]])
        # 2) apply nms
        img_dim = cv2.imread(img_list[ii]).shape  # [height, width, channel
        img_dim = [img_dim[1], img_dim[0]]  # [width, height]
        final_list[str(i.split('/')[-1][:-4])].append(
            [get_valid_box(valid_pred, nms_th, img_dim, model_img_dim, i.split('/')[-1][:-4]), img_dim])

    return final_list


def print_it_on_images(pred_list, print_it=False):
    """
    prints ground truths and selected proposals on image and saves to disk
    """
    # print final predictions and gt's on images
    # class_counter_pr = collections.defaultdict(list)
    # class_counter_gt = collections.defaultdict(list)
    for i, im_name in enumerate(img_list):
        # read image
        img = Image.open(im_name)
        # imgplot = plt.imshow(img)
        fig, ax = plt.subplots()
        ax.imshow(img)
        cur_img_pred = pred_list[str(int(im_name.split('/')[-1].split('.')[0].split('_')[-1]))]

        ###############
        ### predictions
        ###############
        for ii, cls in enumerate(cur_img_pred[0][0].keys()):
            # print(ii, cls, cur_img_pred[0][0][cls][0][1])
            # class_counter_pr[cls]=1
            for kk in cur_img_pred[0][0][cls]:
                x = kk[1][0]
                y = kk[1][1]
                w = kk[1][2] - x
                h = kk[1][3] - y
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        ################
        ### ground-truth
        ################

        cur_img_lbl = open(lbl_list[i], 'r').readlines()
        for k, lbl in enumerate(cur_img_lbl):
            # class_counter_gt[lbl.split(' ')[0]]=1
            kk = nxywhtotlbr([float(i) for i in lbl.split(' ')[1:5]], cur_img_pred[0][1])
            x = kk[0]
            y = kk[1]
            w = kk[2] - x
            h = kk[3] - y
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        if print_it:
            plt.savefig('output/' + str(int(im_name.split('/')[-1].split('.')[0].split('_')[-1])) + '.png')

    return


def assign_gt(pred, gt, iou_th):
    """
    For each GT look for matching prediction, one GT has only 1 TP, rest
    predictions are FP, even if they satisfy iou threshold criteria
    :param pred:
    :param gt:
    :return:
    """
    # initialise all pred with FP - Done
    # for each gt select highest iou for available predictions:
    #   if(iou>0.5): assign that as TP
    for gt_box in gt:
        iou_list = []
        for pred_box in pred:
            cur_iou = iou(pred_box[0], nxywhtotlbr(gt_box, pred_box[3]))
            iou_list.append(cur_iou)
            # pred_box[4] = cur_iou
        if max(iou_list) > iou_th:
            pred[iou_list.index(max(iou_list))][-3] = 'TP'
            pred[iou_list.index(max(iou_list))][-1] = max(iou_list)
    return


#####################
### * Reference * ###
#####################

# for r in recallValues:
#     # Obtain all recall values higher or equal than r
#     argGreaterRecalls = np.argwhere(mrec[:] >= r)
#     pmax = 0
#     # If there are recalls above r
#     if argGreaterRecalls.size != 0:
#         pmax = max(mpre[argGreaterRecalls.min():])
#     recallValid.append(r)
#     rhoInterp.append(pmax)

def get_ap(cls_id, prec_recall_table):
    """
    calculates average precision, taking 11 point on pr-curve
    :param cls_id:
    :param prec_recall_table:
    :return:
    """
    prec = []
    recall = []
    for i in prec_recall_table:
        # print(i, i[0], i[1])
        prec.append(i[0])
        recall.append(i[1])
    gap = 1 / 11
    a = []
    for i in range(0, 11):
        a.append(gap * (i + 1))

    ap = []
    ap_ = []
    for i in a:
        ans = 0
        ans_ = 0
        valid_p = [k for k in recall if k >= i]
        if len(valid_p):
            ans = prec[recall.index([k for k in recall if k >= i][0])]
            ans_ = recall[valid_p.index(valid_p[0])]
        # print(i, ans)
        ap.append(ans)
        ap_.append(ans_)
    print(cls_id, ":", np.average(ap[:-1]))
    return np.average(ap[:-1])


def evaluate(pred_list, iou_th=0.3):
    """
    Calculates average-precision for each class
    :param pred_list: seected proposals from NMS
    :param iou_th: threshold for iou
    :return: AP for each class
    """
    ap_list = collections.defaultdict(list)

    # group predictions from classes together from all images in dataset
    pred = collections.defaultdict(list)
    for i in pred_list.keys():
        for j in pred_list[i][0][0].keys():
            for k in range(0, len(pred_list[i][0][0][j])):
                pred[j].append([i, pred_list[i][0][0][j][k][1], pred_list[i][0][0][j][k][2], pred_list[i][0][1]])

    # group gt from classes together from all images in dataset
    gt = collections.defaultdict(list)
    for i, lbl_file in enumerate(lbl_list):
        lbls = open(lbl_file, 'r').readlines()
        for j, lbl in enumerate(lbls):
            gt[lbl.split(' ')[0]].append([int(lbl_file.split('/')[-1].split('.')[0].split('_')[-1]),
                                          [float(i) for i in lbl.split(' ')[1:5]]])

    # save predictions to be evaluated in convinient format, with flag representing weather it's
    # classified as TP or FP
    pred_upd = collections.defaultdict(list)
    for k in pred.keys():
        t = collections.defaultdict(list)
        for i in pred[k]:
            # box, objectnedd_score, FP/TP, imsize, iou(if TP)
            t[i[0]].append([i[1], i[2], 'FP', i[3], 0])
        pred_upd[k].append(t)

    gt_upd = collections.defaultdict(list)
    total_gt_per_class = collections.defaultdict(list)
    # merge gt of single class into one dict from all images
    for k in gt.keys():
        t = collections.defaultdict(list)
        cnt = 0
        for i in gt[k]:
            t[i[0]].append(i[1])
            cnt += 1
        gt_upd[k].append(t)
        total_gt_per_class[k].append(cnt)
        # for each class get corresponding gt for each prediction in respective class

    pred_final = collections.defaultdict(list)
    prec_recall_list = collections.defaultdict(list)  # all precision-recall values
    # assign corresponding GT to each prediction if IOU is greater than threshold
    # by assigning TP if it's greater than threshold, FP otherwise
    for k_pr in pred_upd.keys():  # class
        for im_pr in pred_upd[k_pr][0].keys():  # image
            assign_gt(pred_upd[k_pr][0][im_pr],
                      gt_upd[str(k_pr)][0][int(im_pr)],
                      iou_th)

        for imgs in pred_upd[k_pr][0]:
            for box in pred_upd[k_pr][0][str(imgs)]:
                # img_name, objectness_score, TP/FP
                pred_final[k_pr].append([imgs, box[1], box[2]])

        # sort based on objectness score
        cur_cls_pred = pred_final[k_pr]
        cur_cls_pred.sort(key=lambda cur_cls_pred: cur_cls_pred[1])
        cur_cls_pred.reverse()
        # --- #TP  #FP  #[TP+FP] #[TP+FN] #Precision #Recall
        # l1   1    0     1       1
        class_stats = [[0, 0, 0, total_gt_per_class[str(k_pr)][0]]]
        class_prec_recall = [[0, 0]]
        # calculate precision and recall
        for i in cur_cls_pred:
            # print(i)
            temp_tpfp = copy.deepcopy(class_stats[-1])
            temp_pr = copy.deepcopy(class_prec_recall[-1])
            if i[2] == 'TP':
                temp_tpfp[0] += 1
                temp_tpfp[2] += 1
            else:
                temp_tpfp[1] += 1
                temp_tpfp[2] += 1
            temp_pr[0] = temp_tpfp[0] / (temp_tpfp[2] + eps)
            temp_pr[1] = temp_tpfp[0] / (temp_tpfp[3] + eps)
            class_stats.append(temp_tpfp)
            class_prec_recall.append(temp_pr)
        prec_recall_list[k_pr].append([class_stats, class_prec_recall])

        # calculate mean Average Precision, using 11 point method
        ap_list[k_pr].append(get_ap(k_pr, class_prec_recall))
    mAP = 0
    for ii in ap_list.keys():
        mAP += ap_list[ii][0]
    mAP /= len(ap_list.keys())
    print('mAP : ', mAP)
    return ap_list


def run():
    print('Run NMS======================================================')
    final_list = apply_nms(prediction_list, nms_th=0.5, objectness_th=0.5, model_img_dim=[416, 416])
    print('Plot GT and final Predictions================================')
    print_it_on_images(final_list, print_it=True)
    print('Evaluate=====================================================')
    mAP = evaluate(final_list, iou_th=0.3)


if __name__ == "__main__":
    run()

print('Done!')
