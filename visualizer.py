import numpy as np

import matplotlib.pyplot as plt


class Visualizer:
    """Bee counter visualizer with blitting
    """
    def __init__(self, sections, n_keep, bins, img_height, arrived_threshold=None, left_threshold=None):
        """
        Args:
            sections (int): number of sections tunnels are split into
            n_keep (int): maximum number of data length kept from previous time steps
            bins (tuple(n*tuple(int, int))): n tunnels' x-axis boundaries on input frames
            img_height (int): expected input frame height
            arrived_threshold (float or None, optional): classification threshold for arrival indicator. Defaults to None.
            left_threshold (float or None, optional): classification threshold for departure indicator. Defaults to None.
        """
        self.rows = sections
        self.cols = len(bins)
        self.x_limit = n_keep
        self.bins = bins
        self.img_height = img_height
        # make figure its axes, number of rows is sections + 1 row for frames
        self.fig, self.axes = plt.subplots(self.rows+1, self.cols)
        self.fig.set_size_inches(18.5, 10.5)
        # first row axes are for frames, rest for time series data
        self.axes = self.axes.reshape(-1)
        self.img_axes = self.axes[:self.cols]
        self.data_axes = self.axes[self.cols:]
        # init all axes
        self.img_data = self.init_img_axes(self.img_axes)
        self.line_data = self.init_data_axes(self.data_axes, arrived_threshold, left_threshold)
        # x-axis data is constant for time series plot, prepare lists for y-axis data
        self.xdata = np.arange(self.x_limit+1)
        self.ydata = [[] for _ in range(self.rows*self.cols)]
        # render figure and save background
        plt.show(block=False)
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        # init bee counters for each tunnel
        self.counters = [ax.text(0.425, 0.075, '', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha='center') for ax in self.img_axes]
    
    def init_img_axes(self, axes):
        """Initiate image viz axes

        Args:
            axes (list(matplotlib.axes.Axes)): axes to draw frames on

        Returns:
            list(matplotlib.image.AxesImage): Data objects on axes
        """
        # maximum tunnel width, matplotlib scales frames to same width if all are not identically wide
        max_bin = max(self.bins, key=lambda x: x[1] - x[0])
        for i, ax in enumerate(axes.reshape(-1)):
            # set axis limits, hide axes, set title
            ax.set_xlim(0, max_bin[1]-max_bin[0])
            ax.set_ylim(0, self.img_height)
            ax.set_axis_off()
            ax.set_title(f'Tunnel {i+1}', loc=('center'))
        # prepare image data objects to write data to
        return [ax.imshow(np.zeros((self.img_height, xbin[1] - xbin[0], 3))) for ax, xbin in zip(axes.reshape(-1), self.bins)]

    def init_data_axes(self, axes, arrived_threshold=None, left_threshold=None):
        """Initiate data viz axes

        Args:
            axes (list(matplotlib.axes.Axes)): axes to plot data on
            arrived_threshold (float or None, optional): classification threshold for arrival indicator. Defaults to None.
            left_threshold (float or None, optional): classification threshold for departure indicator. Defaults to None.

        Returns:
            list(matplotlib.lines.Line2D): Data objects on axes
        """
        for ax in axes.reshape(-1):
            # set axis limits and ticks, hide x-axis labels, set y-axis labels, and turn grid on
            ax.set_xlim(0, self.x_limit)
            ax.set_ylim(-1, 1)
            ax.xaxis.set_ticks(np.arange(0, self.x_limit+1, 5))
            ax.yaxis.set_ticks(np.arange(-1, 2, 1))
            ax.set_xticklabels([])
            ax.tick_params(axis='y', which='major', labelsize=7, pad=0.1)
            ax.grid()
        # prepare plot data objects to write data to
        data_lines = [ax.plot([], [], 'r-')[0] for ax in axes.reshape(-1)]
        # draw classification thresholds
        if arrived_threshold is not None:
            for ax in axes.reshape(-1):
                ax.axhline(y=arrived_threshold, xmin=0, xmax=self.x_limit, linestyle='-', linewidth=0.6)
        if left_threshold is not None:
            for ax in axes.reshape(-1):
                ax.axhline(y=left_threshold, xmin=0, xmax=self.x_limit, linestyle='-', linewidth=0.6)
        return data_lines

    def draw(self, image, data, counters):
        """Draw tunnels and data outputs

        Args:
            image (numpy.ndarray): BGR frame
            data (list(list(float, ...))): data to plot, expected shape of (tunnels, sections)
            counters (list(dict{'up': int, 'down': int})): bee counters of individual tunnels
        """
        # restore background
        self.fig.canvas.restore_region(self.background)
        # draw images and data on specified axes
        self.draw_img_axes(self.img_axes, image, counters)
        self.draw_data_axes(self.data_axes, data)
        # blitting
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def draw_img_axes(self, axes, image, counters):
        """Draw tunnels and counters

        Args:
            axes (list(matplotlib.axes.Axes)): axes to draw tunnels on
            image (numpy.ndarray): BGR frame
            counters (list(dict{'up': int, 'down': int})): bee counters of individual tunnels
        """
        # BGR to RGB
        image = image[:, :, ::-1]
        # set data to axes
        for xbins, data in zip(self.bins, self.img_data):
            data.set_data(image[:, xbins[0]:xbins[1], ...])
        # draw set data
        for ax, data in zip(axes.reshape(-1), self.img_data):
            ax.draw_artist(data)
        # draw bee counters
        for ax, ax_counter, counter in zip(axes, self.counters, counters):
            ax_counter.set_text(f'U:{counter["up"]} D:{counter["down"]}')
            ax.draw_artist(ax_counter)
    
    def draw_data_axes(self, axes, data):
        """Draw tunnels and counters

        Args:
            axes (list(matplotlib.axes.Axes)): axes to plot data on
            data (list(list(float, ...))): data to plot, expected shape of (tunnels, sections)
        """
        for i, tunnel_ratios in enumerate(data):
            for j, section_ratio in enumerate(tunnel_ratios):
                # put input data to corresponding data lists
                curr_list = self.ydata[j*self.cols+i]
                curr_list.append(section_ratio)
                # check for max length
                if len(curr_list) > self.x_limit:
                    del curr_list[:len(curr_list)-self.x_limit-1]
        # set data to axes
        for ln, data in zip(self.line_data, self.ydata):
            ln.set_data(self.xdata[:len(data)], data)
        # draw set data
        for ax, ln in zip(axes.reshape(-1), self.line_data):
            ax.draw_artist(ln)
