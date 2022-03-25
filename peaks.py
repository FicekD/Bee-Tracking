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


class State:
    Arrived = 1
    Left = 2
    Accounted = 3
    Expired = 4


class Track:
    def __init__(self, max_age=5):
        self.state = State.Arrived
        self.max_age = max_age
        self.age = 0

    def update(self):
        self.age += 1
        if self.age > self.max_age:
            self.state = State.Expired
    
    def is_valid(self):
        return self.state == State.Arrived or self.state == State.Left


class Section:
    def __init__(self, n_keep=10, track_max_age=5):
        self.n_keep = n_keep
        self.ratios = list()

        self.track_max_age = track_max_age
        self.tracks = list()

    def update(self, mask):
        self.update_tracks()
        ratio = np.count_nonzero(mask) / mask.size
        self.ratios.append(ratio)
        if len(self.ratios) > self.n_keep:
            self.ratios = self.ratios[len(self.ratios)-self.n_keep:]
        derivative = self.diff()
        if derivative > 0.3:
            cls = 1
            self.tracks.append(Track(max_age=self.track_max_age))
        elif derivative < -0.3:
            cls = -1
            for track in self.tracks:
                if track.state == State.Arrived:
                    track.state = State.Left
        else:
            cls = 0
        return cls    

    def update_tracks(self):
        for track in self.tracks:
            track.update()
        self.tracks = [track for track in self.tracks if track.is_valid()]

    def diff(self, order='first'):
        diff_fn = {
            'first': lambda x: (0 if len(x) < 2 else x[-1] - x[-2]),
            'second': lambda x: (0 if len(x) < 3 else x[-3] - 2*x[-2] + x[-1]),
        }
        result = diff_fn[order](self.ratios)
        return result


class Tunnel:
    def __init__(self, x_boundaries, sections=4, track_max_age=5):
        self.bins = x_boundaries
        self.n_sections = sections

        self.sections = [Section(track_max_age=track_max_age) for _ in range(sections)]
        self.dyn_model = BackgroundModel(50, 50, 30, 5000)

        self.bee_counter = {'up': 0, 'down': 0}

    def update(self, img):
        img = img[:, self.bins[0]:self.bins[1], ...]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.dyn_model.update(gray)
        
        mask = self.dyn_model.get_mask(gray)
        splits = np.split(mask, self.n_sections, axis=0)
        classes = [section.update(split) for section, split in zip(self.sections, reversed(splits))]
            
        self.assign_tracks()
        return classes
    
    def assign_tracks(self):
        tracks_to_assign = list()
        for section in self.sections:
            for track in section.tracks:
                if track.state == State.Left:
                    tracks_to_assign.append(track)
                    break
            else:
                return
        for track in tracks_to_assign:
            track.state = State.Accounted
        key = 'up' if tracks_to_assign[0].age < tracks_to_assign[-1].age else 'down'
        self.bee_counter[key] += 1


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
        
        self.counters = [ax.text(0.425, 0.1, "elp", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center") for ax in self.img_axes]
    
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

    def draw(self, image, ratios, counters):
        self.fig.canvas.restore_region(self.background)

        self.draw_img_axes(self.img_axes, image, counters)
        self.draw_data_axes(self.data_axes, ratios)
        
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def draw_img_axes(self, axes, image, counters):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for xbins, data in zip(self.bins, self.img_data):
            data.set_data(image[:, xbins[0]:xbins[1], ...])
        for ax, data in zip(axes.reshape(-1), self.img_data):
            ax.draw_artist(data)
        for ax, ax_counter, counter in zip(axes, self.counters, counters):
            ax_counter.set_text(f'U:{counter["up"]} D:{counter["down"]}')
            ax.draw_artist(ax_counter)
    
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
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)

    bins = (
        (80, 150), (175, 217), (230, 300), (327, 384),
        (407, 471), (490, 560), (570, 650), (663, 724),
        (749, 805), (833, 893), (913, 978), (987, 1046)
    )
    sections = 4
    tunnels = [Tunnel(xbin, sections=sections, track_max_age=15) for xbin in bins]

    viz = Visualizer(sections, 50, bins, 140)

    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)[20:, ...]

        ratios = [tunnel.update(img) for tunnel in tunnels]
        counters = [tunnel.bee_counter for tunnel in tunnels]
    
        viz.draw(img, ratios, counters)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
