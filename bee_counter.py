"""
Bee counter from files

<dir>
├── data
│  └── 210906_Pokus2
│     └── *.jpg
│  └── 210906_Pokus2_sorted
│     └── *.jpg
├── background.py
├── bee_counter.py
├── tracker.py
├── visualizer.py
"""

__authors__ = ['Bc. Dominik Ficek', 'Bc. David Makówka']
__credits__ = ['Ing. Šimon Bilík']

import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

from time import time

from tracker import Tunnel
from visualizer import Visualizer


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
    arrived_threshold = 0.3
    left_threshold = -0.3
    track_max_age = 20
    tunnels = [Tunnel(xbin, sections=sections, track_max_age=track_max_age, arrived_threshold=arrived_threshold, left_threshold=left_threshold) for xbin in bins]

    frame_heght = 140
    x_axis_len = 25
    viz = Visualizer(sections, x_axis_len, bins, frame_heght, arrived_threshold, left_threshold)

    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)

        s = time()
        data = [tunnel.update(img[20:, ...]) for tunnel in tunnels]
        update_time = time() - s
        counters = [tunnel.bee_counter for tunnel in tunnels]
        
        sys.stdout.write(f'\rUpdate latency: {update_time*1e3:.2f}ms{10*" "}')
        sys.stdout.flush()
    
        viz.draw(img, data, counters)
        plt.pause(0.2)
        # plt.waitforbuttonpress()
        # plt.savefig(os.path.join(base_path, 'data', f'{dataset}_results', file))


if __name__ == '__main__':
    main()
