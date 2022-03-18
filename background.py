import os

import numpy as np
import cv2

import matplotlib.pyplot as plt


def img_generator(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        img_path = os.path.join(dir_path, file)
        img = cv2.imread(img_path)
        yield img


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

    @staticmethod
    def is_dynamic(img1, img2, diff_th, count_th):
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        return np.count_nonzero(diff > diff_th) > count_th


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2'
    dataset_path = os.path.join(base_path, 'data', dataset)

    dyn_model = BackgroundModel(50, 50, 30, 2000)
    for img in img_generator(dataset_path):
        img = cv2.medianBlur(img, 7)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dyn_model.update(gray)
        # if prev_gray is None:
        #     prev_gray = gray
        #     model = gray.astype(np.float32)
        #     continue

        # dynamic_scene = is_dynamic(gray, prev_gray, diff_th, count_diff_th)
        # if not dynamic_scene:
        #     dynamic_model = is_dynamic(gray, model, model_diff_th, model_count_diff_th)
        #     if not dynamic_model:
        #         print('updating')
        #         model = alpha * gray.astype(np.float32) + (1 - alpha) * model

        model_img = dyn_model.model.astype(np.uint8)
        # _, thresholded = cv2.threshold(hsv[int(hsv.shape[0]*0.3):int(hsv.shape[0]*0.7), :, 2], 70, 255, cv2.THRESH_BINARY)
        # thresholded = cv2.adaptiveThreshold(hsv[int(hsv.shape[0]*0.3):int(hsv.shape[0]*0.7), :, 2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)
        # edges = cv2.Canny(model_img[int(gray.shape[0]*0.3):int(gray.shape[0]*0.7), :], 40, 130)

        # b, bins, patches = plt.hist(gray.reshape(-1), 255)
        # plt.xlim([0,255])
        # plt.show()

        # mask = np.abs(gray.astype(np.float32) - model.astype(np.float32)) > diff_th
        # masked = img * mask[..., np.newaxis]

        cv2.imshow('img', img)
        # cv2.imshow('thresholded', edges)
        # cv2.imshow('masked', masked)
        cv2.imshow('model', model_img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
