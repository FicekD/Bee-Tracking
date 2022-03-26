import numpy as np
import cv2


class BackgroundModel:
    """Background subtraction adaptive model
    """
    def __init__(self, diff_th, count_diff_th, model_diff_th, model_count_diff_th, alpha=0.01):
        """
        Args:
            diff_th (int): difference threshold to classify scene as dynamic or static
            count_diff_th (int): number of dynamic pixels required to classify as dynamic
            model_diff_th (int): difference threshold to classify frame as dynamic with respect to model
            model_count_diff_th (int): number of dynamic pixels required to classify scene as dynamic with respect to model
            alpha (float, optional): model learning rate. Defaults to 0.01.
        """
        self.diff_th = diff_th
        self.count_diff_th = count_diff_th
        self.model_diff_th = model_diff_th
        self.model_count_diff_th = model_count_diff_th
        self.alpha = alpha

        self.prev = None
        self.model = None

    def update(self, img):
        """Update dynamic model, first frame is set as the initial model

        Args:
            img (numpy.ndarray): monochromatic frame
        """
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
        """Motion mask, frame to model

        Args:
            img (np.ndarray): monochromatic frame

        Returns:
            numpy.ndarray: boolean mask
        """
        mask = np.abs(img.astype(np.float32) - self.model.astype(np.float32)) > self.model_diff_th
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7))).astype(bool)
        return mask

    @staticmethod
    def is_dynamic(img1, img2, diff_th, count_th):
        """Classify scene as dynamic or static from two frames

        Args:
            img1 (numpy.ndarray): monochromatic frame
            img2 (numnpy.ndarray): monochromatic frame
            diff_th (int): difference threshold to classify scene as dynamic or static
            count_th (int): number of dynamic pixels required to classify as dynamic

        Returns:
            bool: True if dynamic, False otherwise
        """
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        return np.count_nonzero(diff > diff_th) > count_th
