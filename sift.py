import os

import numpy as np
import cv2


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)

    sift = cv2.SIFT_create()

    binx = (325, 380)
    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)
        img = img[:, binx[0]:binx[1], :]
        # img = cv2.GaussianBlur(img, (3, 3), 1.6)
        img = cv2.medianBlur(img, 7)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = sift.detect(gray, None)

        cv2.drawKeypoints(img, features, img)
        cv2.line(img, (binx[0], 0), (binx[0], img.shape[0]), (255, 0, 0), 2)
        cv2.line(img, (binx[1], 0), (binx[1], img.shape[0]), (255, 0, 0), 2)


        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
