import numpy as np


class ChannelStandardization(object):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, data):
        mean = np.mean(data, axis=self.axis, keepdims=True)
        std = np.std(data, axis=self.axis, keepdims=True)
        new_data = (data - mean) / std
        return new_data


class MapLabel(object):
    def __init__(self):
        pass

    def __call__(self, target):
        # TODO: this is a stupid transform, just as an example
        if target < 10:
            new_target = 0
        else:
            new_target = 1
        return new_target