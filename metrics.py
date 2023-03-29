import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.metrics import hausdorff_distance
from typing import Optional, Dict


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


def compute_panoptic_quality(matches, n_classes: int = 4) -> Dict:
    unmatched_gt = list(matches.gt_idxs_class.keys())
    unmatched_pred = list(matches.pred_idxs_class.keys())

    TPc = [0 for _ in range(n_classes)]
    FPc = [0 for _ in range(n_classes)]
    FNc = [0 for _ in range(n_classes)]
    IoUc = [[] for _ in range(n_classes)]

    for match in matches.matches:
        if match.iou > 0.5 and matches.gt_idxs_class[match.gt_idx] == matches.pred_idxs_class[match.pred_idx]:
            unmatched_gt.remove(match.gt_idx)
            unmatched_pred.remove(match.pred_idx)
            TPc[matches.gt_idxs_class[match.gt_idx] - 1] += 1
            IoUc[matches.gt_idxs_class[match.gt_idx] - 1].append(match.iou)

    for gt_idx in unmatched_gt:
        FNc[matches.gt_idxs_class[gt_idx] - 1] += 1

    for pred_idx in unmatched_pred:
        FPc[matches.pred_idxs_class[pred_idx] - 1] += 1

    SQc = []
    RQc = []
    PQc = []
    for c in range(n_classes):
        if TPc[c]+FPc[c]+FNc[c] == 0:
            continue
        RQ = 2 * TPc[c] / (2 * TPc[c] + FPc[c] + FNc[c])
        if len(IoUc[c]) == 0:
            SQ = 0
        else:
            SQ = np.mean(IoUc[c])
        RQc.append(RQ)
        SQc.append(SQ)
        PQc.append(RQ * SQ)
    return {
        "PQ": np.mean(PQc),
        "PQc": PQc,
        "SQc": SQc,
        "RQc": RQc,
        "TPc": TPc,
        "FPc": FPc,
        "FNc": FNc,
        "IoUc": IoUc
    }
