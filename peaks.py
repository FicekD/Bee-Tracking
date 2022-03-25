import os

import numpy as np
import cv2

import matplotlib.pyplot as plt


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


class Section:
    def __init__(self, n_keep=50):
        self.n_keep = n_keep
        self.ratios = list()

    def update(self, mask):
        ratio = np.count_nonzero(mask) / mask.size
        self.ratios.append(ratio)
        if len(self.ratios) > self.n_keep:
            self.ratios = self.ratios[len(self.ratios)-self.n_keep:]

    def diff(self):
        return 0 if len(self.ratios) < 2 else self.ratios[-1] - self.ratios[-2]


class TunnelProcessor:
    def __init__(self, x_boundaries, sections=4):
        self.bins = x_boundaries
        self.n_sections = sections
        self.sections = [Section() for _ in range(sections)]
        self.dyn_model = BackgroundModel(50, 50, 30, 5000)
    
    def update(self, img):
        img = img[:, self.bins[0]:self.bins[1], ...]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.dyn_model.update(gray)
        
        mask = self.dyn_model.get_mask(gray)
        splits = np.split(mask, self.n_sections, axis=0)
        for section, split in zip(self.sections, splits):
            section.update(split)
    
    def diffs(self):
        return [section.diff() for section in self.sections]


class Visualizer:
    def __init__(self, sections, x_limit, bins, img_height):
        self.rows = sections
        self.cols = len(bins)
        self.x_limit = x_limit
        self.bins = bins
        self.img_height = img_height

        self.fig, self.axes = plt.subplots(self.rows+1, self.cols)
        self.fig.set_size_inches(18.5, 10.5)

        self.axes = self.axes.reshape(-1)
        self.img_axes = self.axes[:self.cols]
        self.data_axes = self.axes[self.cols:]

        self.img_data = self.init_img_axes(self.img_axes)
        self.line_data = self.init_data_axes(self.data_axes)
        
        self.xdata = np.arange(self.x_limit+1)
        self.ydata = [[] for _ in range(self.rows*self.cols)]

        plt.show(block=False)
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
    
    def init_img_axes(self, axes):
        max_bin = max(self.bins, key=lambda x: x[1] - x[0])
        for i, ax in enumerate(axes.reshape(-1)):
            ax.set_xlim(0, max_bin[1]-max_bin[0])
            ax.set_ylim(0, self.img_height)
            ax.set_axis_off()
            ax.set_title(f'Tunnel {i+1}', loc=('center'))
        return [ax.imshow(np.zeros((self.img_height, xbin[1] - xbin[0], 3))) for ax, xbin in zip(axes.reshape(-1), self.bins)]

    def init_data_axes(self, axes):
        for ax in axes.reshape(-1):
            ax.set_xlim(0, self.x_limit)
            ax.set_ylim(-1, 1)
            ax.xaxis.set_ticks(np.arange(0, self.x_limit+1, 5))
            ax.yaxis.set_ticks(np.arange(-1, 2, 1))
            ax.set_xticklabels([])
            ax.tick_params(axis='y', which='major', labelsize=7, pad=0.1)
            ax.grid()
        return [ax.plot([], [], 'r-')[0] for ax in axes.reshape(-1)]

    def draw(self, image, ratios):
        self.fig.canvas.restore_region(self.background)

        self.draw_img_axes(self.img_axes, image)
        self.draw_data_axes(self.data_axes, ratios)
        
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def draw_img_axes(self, axes, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for xbins, data in zip(self.bins, self.img_data):
            data.set_data(image[:, xbins[0]:xbins[1], ...])
        for ax, data in zip(axes.reshape(-1), self.img_data):
            ax.draw_artist(data)
    
    def draw_data_axes(self, axes, data):
        for i, tunnel_ratios in enumerate(data):
            for j, section_ratio in enumerate(tunnel_ratios):
                curr_list = self.ydata[j*self.cols+i]
                curr_list.append(section_ratio)
                if len(curr_list) > self.x_limit:
                    del curr_list[:len(curr_list)-self.x_limit-1]
        for ln, data in zip(self.line_data, self.ydata):
            ln.set_data(self.xdata[:len(data)], data)
        for ax, ln in zip(axes.reshape(-1), self.line_data):
            ax.draw_artist(ln)


def main():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)

    bins = (
        (80, 150), (175, 217), (230, 300), (327, 384),
        (407, 471), (490, 560), (570, 650), (663, 724),
        (749, 805), (833, 893), (913, 978), (987, 1046)
    )
    sections = 4
    processors = [TunnelProcessor(xbin, sections=sections) for xbin in bins]

    viz = Visualizer(sections, 50, bins, 140)

    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)[20:, ...]

        ratios = list()
        for processor in processors:
            processor.update(img)
            ratios.append(processor.diffs())
    
        viz.draw(img, ratios)


if __name__ == '__main__':
    main()
