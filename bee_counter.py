"""
Bee counter from files

<dir>
├── data
│  └── 210906_Pokus2
│     └── *.jpg
│  └── 210906_Pokus2_sorted
│     └── *.jpg
├── bee_counter.ini
├── background.py
├── bee_counter.py
├── tracker.py
├── threads.py
├── visualizer.py
"""

__authors__ = ['Bc. Dominik Ficek', 'Bc. David Makówka']
__credits__ = ['Ing. Šimon Bilík']

import os
import sys

import configparser
import logging
import json
import cv2
import matplotlib.pyplot as plt

from time import time, sleep
from functools import partial

from tracker import Tunnel
from visualizer import Visualizer

from threads import ProducerThread, ConsumerThread, _run_event


def standalone():
    base_path = os.path.dirname(os.path.realpath(__file__))
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)
    files = os.listdir(dataset_path)

    cfg_path = os.path.join(base_path, 'bee_counter.ini')
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    bins = json.loads(cfg.get('ImageProcessing', 'bins'))
    sections = cfg.getint('ImageProcessing', 'sections')
    arrived_threshold = cfg.getfloat('ImageProcessing', 'arrived_threshold')
    left_threshold = cfg.getfloat('ImageProcessing', 'left_threshold')
    track_max_age = cfg.getint('ImageProcessing', 'track_max_age')
    background_init_from_file = cfg.getboolean('ImageProcessing', 'background_init_from_file')

    background_init_frame = None
    if background_init_from_file:
        background_init_frame = cv2.imread(os.path.join(base_path, 'data', 'background.jpg'))

    tunnels = [Tunnel(xbin, sections=sections, track_max_age=track_max_age, arrived_threshold=arrived_threshold, left_threshold=left_threshold, background_init_frame=background_init_frame) for xbin in bins]

    frame_heght = 140
    x_axis_len = 25
    viz = Visualizer(sections, x_axis_len, bins, frame_heght, arrived_threshold, left_threshold)

    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = cv2.imread(img_path)

        s = time()
        data = [tunnel.update(img) for tunnel in tunnels]
        update_time = time() - s
        counters = [tunnel.bee_counter for tunnel in tunnels]
        
        sys.stdout.write(f'\rUpdate latency: {update_time*1e3:.2f}ms     ')
        sys.stdout.flush()
    
        viz.draw(img, data, counters)
        plt.pause(0.2)
        # plt.waitforbuttonpress()
        # plt.savefig(os.path.join(base_path, 'data', f'{dataset}_results', file))


def threaded_config():
    logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-5s) %(message)s')

    base_path = os.path.dirname(os.path.realpath(__file__))
    
    dataset = '210906_Pokus2_sorted'
    dataset_path = os.path.join(base_path, 'data', dataset)

    cfg_path = os.path.join(base_path, 'bee_counter.ini')
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    sections = cfg.getint('ImageProcessing', 'sections')
    arrived_threshold = cfg.getfloat('ImageProcessing', 'arrived_threshold')
    left_threshold = cfg.getfloat('ImageProcessing', 'left_threshold')
    track_max_age = cfg.getint('ImageProcessing', 'track_max_age')
    background_init_from_file = cfg.getboolean('ImageProcessing', 'background_init_from_file')
    if background_init_from_file:
        background_init_frame = cv2.imread(os.path.join(base_path, 'data', 'background.jpg'))
    tunnel_func = partial(Tunnel, sections=sections, track_max_age=track_max_age, arrived_threshold=arrived_threshold, left_threshold=left_threshold, background_init_frame=background_init_frame)
    
    tunnel_args = json.loads(cfg.get('ImageProcessing', 'bins'))

    producer = ProducerThread(dataset_path, 'producer')
    consumer = ConsumerThread(tunnel_func, tunnel_args, 'consumer')

    producer.start()
    consumer.start()

    try:
        while _run_event.is_set():
            sleep(0.1)
    except KeyboardInterrupt:
        _run_event.clear()
        producer.join()
        consumer.join()


if __name__ == '__main__':
    # standalone()
    threaded_config()
