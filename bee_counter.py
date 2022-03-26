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

import numpy as np
import cv2

import matplotlib.pyplot as plt

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
    tunnels = [Tunnel(xbin, sections=sections, track_max_age=20) for xbin in bins]

    frame_heght = 120
    x_axis_len = 50
    viz = Visualizer(sections, x_axis_len, bins, frame_heght)

    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)[20:, ...]

        data = [tunnel.update(img) for tunnel in tunnels]
        counters = [tunnel.bee_counter for tunnel in tunnels]
    
        viz.draw(img, data, counters)
        plt.pause(0.2)
        # plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
