import numpy as np
from terminaltables import AsciiTable

from sklearn.metrics import confusion_matrix

def print_metrics(metrics, samples, phase, epoch):
    metric_keys = list(metrics.keys())
    metric_vals = ["{:6f}".format(metrics[i]/samples) for i in metric_keys]
    metric_keys = ["epoch"] + ["phase"] + metric_keys
    metric_vals = [f"{epoch}"] + [f"{phase}"] + metric_vals
    metric_table = [metric_keys, metric_vals]
    print (AsciiTable(metric_table).table)


def dice_coeff(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    # im1 = np.asarray(im1).astype(np.bool)
    # im2 = np.asarray(im2).astype(np.bool)
    im1 = im1 > 0.5
    im2 = im2 > 0.5

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())
