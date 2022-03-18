import os

import numpy as np
import cv2


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)
    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)
        # img = cv2.GaussianBlur(img, (3, 3), 1.6)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, 20, 0.7, 10)

        for feature in features.reshape(-1, 2):
            cv2.circle(img, tuple(feature.astype(np.int32)), 1, (255, 0, 0), -1)

        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
