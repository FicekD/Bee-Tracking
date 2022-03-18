import os

import numpy as np
import cv2

from itertools import chain, zip_longest


class BackgroundModel:
    def __init__(self, diff_th, count_diff_th, model_diff_th, model_count_diff_th, alpha=0.01):
        self.diff_th = diff_th
        self.count_diff_th = count_diff_th
        self.model_diff_th = model_diff_th
        self.model_count_diff_th = model_count_diff_th
        self.alpha = alpha

        self.prev = None
        self.model = None

    def update(self, img):
        if self.prev is None or self.model is None:
            self.prev, self.model = img, img
            return
        dynamic_scene = self.is_dynamic(img, self.prev, self.diff_th, self.count_diff_th)
        if not dynamic_scene:
            dynamic_model = self.is_dynamic(img, self.model, self.model_diff_th, self.model_count_diff_th)
            if not dynamic_model:
                self.model = self.alpha * img.astype(np.float32) + (1 - self.alpha) * self.model
        self.prev = img
    
    def get_mask(self, img):
        return np.abs(img.astype(np.float32) - self.model.astype(np.float32)) > self.model_diff_th


    @staticmethod
    def is_dynamic(img1, img2, diff_th, count_th):
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        return np.count_nonzero(diff > diff_th) > count_th


class TunnelProcessor:
    def __init__(self, x_boundaries):
        self.bins = x_boundaries
        self.sift = cv2.SIFT_create()
        self.dyn_model = BackgroundModel(50, 50, 30, 5000)
    
    def process(self, img):
        img = img[:, self.bins[0]:self.bins[1], ...]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.dyn_model.update(gray)
        features = self.sift.detect(gray, None)
        
        mask = self.dyn_model.get_mask(gray)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7))).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15))).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, np.ones((5, 5))).astype(bool)
        features = [feature for feature in features if mask[tuple((round(x) for x in reversed(feature.pt)))]]

        # cv2.drawKeypoints(img, features, img)
        return img


def join_imgs(imgs):
    line = np.full((imgs[0].shape[0], 2, 3), (0, 0, 255), dtype=np.uint8)
    img = np.concatenate(list(chain(*zip_longest(imgs, [], fillvalue=line))), axis=1)
    return img


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)

    bins = (
        (80, 150), (175, 217), (230, 300), (327, 384),
        (407, 471), (490, 560), (663, 724), (749, 805),
        (833, 893), (913, 978), (987, 1046)
    )
    processors = [TunnelProcessor(xbin) for xbin in bins]
    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)

        imgs = [processor.process(img) for processor in processors]
        models = [cv2.cvtColor(processor.dyn_model.model.astype(np.uint8), cv2.COLOR_GRAY2BGR) for processor in processors]

        combined = join_imgs(imgs)
        models = join_imgs(models)
        cv2.imshow('img', img)
        cv2.imshow('combined', combined)
        cv2.imshow('models', models)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
