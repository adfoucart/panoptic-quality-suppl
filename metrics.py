import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.metrics import hausdorff_distance
from typing import Optional


def compute_iou(a: np.array, b: np.array, none_value: Optional[float] = None) -> Optional[float]:
    """Computes the IoU between the binary masks a and b"""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    if (a + b).sum() == 0:
        return none_value
    return (a * b).sum() / (a + b).sum()


def compute_hd(a: np.array, b: np.array, percentile: Optional[float] = None) -> float:
    """Computes Hausdorff's distance between the contours of two binary masks.
    Optionally, the percentile version of the HD can be used (for instance, HD95."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    c_a = (edt(a) == 1)
    c_b = (edt(b) == 1)

    if percentile is not None:
        dc_a = edt(c_a == 0)[c_b]
        dc_b = edt(c_b == 0)[c_a]

        n_a = int(percentile * len(dc_a))
        n_b = int(percentile * len(dc_b))

        return max(np.sort(dc_a.flatten())[n_a], np.sort(dc_b.flatten())[n_b])

    return hausdorff_distance(c_a, c_b)
