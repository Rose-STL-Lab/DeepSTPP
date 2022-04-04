import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def normalize_color(color):
    r, g, b = color
    return (r / 255.0, g / 255.0, b / 255.0)


def color_list(colors):
    res = []
    N = len(colors)
    for i in range(N - 1):
        res.append(normalize_color(colors[i]))
        res.append(normalize_color(colors[i+1]))
        res.append((i + 1) * 1.0 / (N - 1))
    return res[:-1]


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


rose_muted = make_colormap(color_list([
                (24, 51, 46),
                (73, 121, 107),
                (169, 186, 157),
                (208, 123, 135),
                (183, 110, 120),
                (165, 98, 105),
                (235, 197, 201)]))
rose_vivid = make_colormap(color_list([
                (24, 51, 46),
                (166, 214, 8),
                (107, 161, 42),
                (203, 0, 44),
                (227, 20, 86),
                (255, 0, 126),
                (235, 197, 201)]))
rose = make_colormap(color_list([
                (100, 182, 75),
                (151, 200, 118),
                (182, 212, 149),
                (194, 30, 86),
                (213, 55, 107),
                (157, 23, 69)]))
