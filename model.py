# Base on Repo of Zylo117@git

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_raw, invert_affine, postprocess


class Model():
    def __init__(self, force_input_size=512, threshold=0.2, iou_threshold=0.2):
        self.force_input_size = force_input_size  # set None to use default size

        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.compound_coef = 0
        self.use_cuda = True
        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True

        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                         'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                         'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                         'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                         'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                         'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                         'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                         'toothbrush']

        # tf bilinear interpolation is different from any other's, just make do
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = self.input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size

        self.model = EfficientDetBackbone(
            compound_coef=self.compound_coef, num_classes=len(self.obj_list))
        self.model.load_state_dict(torch.load(
            f'weights/efficientdet-d{self.compound_coef}.pth'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()


    def predict(self, raw_img):
        self.ori_imgs, self.framed_imgs, self.framed_metas = preprocess_raw(raw_img, max_size=self.input_size)
        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in self.framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in self.framed_imgs], 0)
        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            self.features, self.regression, self.classification, self.anchors = self.model(x)

            self.regressBoxes = BBoxTransform()
            self.clipBoxes = ClipBoxes()

            out = postprocess(x,
                            self.anchors, self.regression, self.classification,
                            self.regressBoxes, self.clipBoxes,
                            self.threshold, self.iou_threshold)
            pred = invert_affine(self.framed_metas, out)
            return pred



    def label_img(self, preds, imgs):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue

            for j in range(len(preds[i]['rois'])):
                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
        
        return imgs
    

    def run(self, raw_img):
        pred_label = self.predict(raw_img)
        pred_img = self.label_img(pred_label, self.ori_imgs)
        return pred_img[0]

