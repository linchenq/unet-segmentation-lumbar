import numpy as np
from terminaltables import AsciiTable

def print_metrics(metrics, samples, phase, epoch):
    metric_keys = list(metrics.keys())
    metric_vals = ["{:6f}".format(metrics[i]/samples) for i in metric_keys]
    metric_keys = ["epoch"] + ["phase"] + metric_keys
    metric_vals = [f"{epoch}"] + [f"{phase}"] + metric_vals
    metric_table = [metric_keys, metric_vals]
    print (AsciiTable(metric_table).table)

def dice_coeff(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum