import cv2
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from argparse import ArgumentParser
from model import Model

if __name__ == "__main__":
    src_dir = './test_images/'
    dst_dir = './test_images/output/'
    ext_name = 'jpg'

    model = Model(compound_coef=2, threshold=0.1, iou_threshold=0.1)
    for file in tqdm(os.listdir(src_dir)):
        if file.endswith('jpg'):
            filename = file.split('.')[0]
            raw_img = cv2.imread(src_dir + file)
            pred_img = model.run(raw_img)
            cv2.imwrite(dst_dir + filename + '.jpg', pred_img)
