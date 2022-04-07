import os

import logging
import threading
import queue
import time
import numpy as np

from PIL import Image


BUFF_SIZE = 10
_queue = queue.Queue(BUFF_SIZE)
_run_event = threading.Event()
_run_event.set()


class ProducerThread(threading.Thread):
    def __init__(self, root_path, name):
        super().__init__()
        self.root_path = root_path
        self.files = os.listdir(root_path)
        self.name = name

    def run(self):
        for file in self.files:
            if not _run_event.is_set():
                break
            img = Image.open(os.path.join(self.root_path, file))
            if not _queue.full():
                _queue.put(img)
                logging.debug(f'Putting img : {str(_queue.qsize())} items in queue')
                time.sleep(0.05)
            else:
                logging.warning(f'Skipped frame')
        _run_event.clear()


class ConsumerThread(threading.Thread):
    def __init__(self, processor_class, processor_args, name):
        super().__init__()
        self.processors = [processor_class(arg) for arg in processor_args]
        self.counters = [processor.bee_counter for processor in self.processors]
        self.name = name

    def run(self):
        while _run_event.is_set():
            if _queue.empty():
                time.sleep(0.01)
                continue
            img = _queue.get()
            logging.debug(f'Getting img : {str(_queue.qsize())} items in queue')
            # RGB -> BGR
            img = np.asarray(img)[..., ::-1]
            for processor in self.processors:
                processor.update(img)
