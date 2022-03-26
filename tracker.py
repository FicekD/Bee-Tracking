import numpy as np
import cv2

from background import BackgroundModel


class State:
    """Track states
    """
    Arrived = 1
    Left = 2
    Accounted = 3
    Expired = 4


class Track:
    """Single bee track
    """
    def __init__(self, max_age=5):
        """
        Args:
            max_age (int, optional): Maximum pending age, track is dismissed after reaching its max age. Defaults to 5.
        """
        self.state = State.Arrived
        self.max_age = max_age
        self.age = 0

    def update(self):
        """Update track age
        """
        self.age += 1
        if self.age > self.max_age:
            self.state = State.Expired
    
    def is_valid(self):
        """Track is valid if it's not expired or already accounted for

        Returns:
            bool: validity flag
        """
        return self.state == State.Arrived or self.state == State.Left


class Section:
    """Tunnel section
    """
    def __init__(self, n_keep=10, track_max_age=5):
        """
        Args:
            n_keep (int, optional): number of kept information from previous time steps. Defaults to 10.
            track_max_age (int, optional): maximum track age, refer to Track. Defaults to 5.
        """
        self.n_keep = n_keep
        self.ratios = list()

        self.track_max_age = track_max_age
        self.tracks = list()

    def update(self, mask):
        """Update section from motion mask

        Args:
            mask (numpy.ndarray): motion boolean mask

        Returns:
            float: output class or derivative for viz purposes
        """
        self.update_tracks()
        ratio = np.count_nonzero(mask) / mask.size
        self.ratios.append(ratio)
        if len(self.ratios) > self.n_keep:
            self.ratios = self.ratios[len(self.ratios)-self.n_keep:]
        derivative = self.diff()
        if derivative > 0.3:
            cls = 1
            self.tracks.append(Track(max_age=self.track_max_age))
        elif derivative < -0.1:
            cls = -1
            for track in self.tracks:
                if track.state == State.Arrived:
                    track.state = State.Left
        else:
            cls = 0
        # return cls  
        return derivative    

    def update_tracks(self):
        """Update all existing tracks
        """
        for track in self.tracks:
            track.update()
        self.tracks = [track for track in self.tracks if track.is_valid()]

    def diff(self, order='first'):
        """Calculate differentiation from last valid time step

        Args:
            order (str, optional): Order of diff, either "first" or "second". Defaults to 'first'.

        Returns:
            float: diff result
        """
        diff_fn = {
            'first': lambda x: (0 if len(x) < 2 else x[-1] - x[-2]),
            'second': lambda x: (0 if len(x) < 3 else x[-3] - 2*x[-2] + x[-1]),
        }
        result = diff_fn[order](self.ratios)
        return result


class Tunnel:
    """Tunnel
    """
    def __init__(self, x_boundaries, sections=4, track_max_age=5):
        """s:
            x_boundaries (tuple(int, int)): tunnel's x-axis boundaries on input frames
            sections (int, optional): number of sections the tunnel will be split on along the y-axis. Defaults to 4.
            track_max_age (int, optional): maximum track age, refer to Track. Defaults to 5.
        """
        self.bins = x_boundaries
        self.n_sections = sections

        self.sections = [Section(track_max_age=track_max_age) for _ in range(sections)]
        self.dyn_model = BackgroundModel(50, 50, 30, 5000)

        self.bee_counter = {'up': 0, 'down': 0}

    def update(self, img):
        """Update on next time step

        Args:
            img (numpy.ndarray): BGR frame

        Returns:
            list: list of outputs from individual section updates, for viz purposes
        """
        img = img[:, self.bins[0]:self.bins[1], ...]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.dyn_model.update(gray)
        
        mask = self.dyn_model.get_mask(gray)
        splits = np.split(mask, self.n_sections, axis=0)
        outputs = [section.update(split) for section, split in zip(self.sections, reversed(splits))]
            
        self.assign_tracks()
        return outputs
    
    def assign_tracks(self):
        """Try to assign tracks from sections to bee traveling upwards or downwards
        """
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
