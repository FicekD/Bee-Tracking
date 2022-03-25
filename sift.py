import os

import numpy as np
import cv2

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
        img = cv2.medianBlur(img, 7)
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
        # img = cv2.GaussianBlur(img, (5, 5), 1.6)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.dyn_model.update(gray)
        features = self.sift.detect(gray, None)
        
        mask = self.dyn_model.get_mask(gray)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7))).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15))).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_DILATE, np.ones((3, 3))).astype(bool)
        features = [feature for feature in features if mask[tuple((round(x) for x in reversed(feature.pt)))]]

        points = [(feature.pt[0] + self.bins[0], feature.pt[1]) for feature in features]
        points = np.array(points).reshape(-1, 2)
        if points.shape[0] < 5:
            return []

        best_score = -1e6
        for k in range(1, 4):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(points)
            score = -kmeans.inertia_ * k ** 2
            # if k == 1:
            #     score = kmeans.inertia_ / 1e4
            # else:
            #     score = silhouette_score(points, kmeans.labels_) / k
            if score > best_score:
                best_score = score
                model = kmeans

        labels = model.labels_
        labeled_points = list()
        for label in np.unique(labels):
            points_l = points[labels == label, :]
            labeled_points.append(points_l.astype(np.int32).tolist())
        return labeled_points


def draw(img, points):
    for points_c in points:
        color = tuple(np.random.uniform(0, 255, (3, )).astype(np.int32).tolist())
        for point in points_c:
            cv2.circle(img, point, 2, color, -1)


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2'
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
        img = cv2.imread(img_path)[20:, ...]

        points = list()
        for processor in processors:
            points += processor.process(img)
    
        draw(img, points)

        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
