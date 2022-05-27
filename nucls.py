import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from skimage.draw import polygon
from tqdm import tqdm
import csv
from skimage.measure import regionprops
from metrics import compute_iou, compute_hd


@dataclass
class Annotation:
    raw_classification: str
    main_classification: str
    super_classification: str
    coords_x: List[int]
    coords_y: List[int]


def load_annotations(path: str) -> Tuple[Dict[str, Dict[str, List[Annotation]]], List[str]]:
    annotations = {}
    all_slides = []

    files = os.listdir(path)

    for f in tqdm(files):
        parts = f.split('_')
        pathologist = parts[0]
        if pathologist.startswith('NP'):  # non-pathologist
            continue
        slide = parts[4]
        if slide not in all_slides:
            all_slides.append(slide)
        left = int(parts[6].split('-')[1])
        top = int(parts[7].split('-')[1])
        if pathologist not in annotations:
            annotations[pathologist] = {}
        if slide not in annotations[pathologist]:
            annotations[pathologist][slide] = []

        with open(os.path.join(path, f), 'r') as fp:
            reader = csv.reader(fp)
            header = None
            for row in reader:
                if header is None:
                    header = row
                    continue
                if row[4] != 'polyline':
                    continue
                anno = Annotation(
                    raw_classification=row[1],
                    main_classification=row[2],
                    super_classification=row[3],
                    coords_x=[int(x) + left for x in row[9].split(',')],
                    coords_y=[int(y) + top for y in row[10].split(',')]
                )
                annotations[pathologist][slide].append(anno)

    return annotations, all_slides


@dataclass
class Match:
    area: int
    iou: float
    hd: float


def match_experts(annotations: Dict, all_slides: List[str]) -> List[Match]:
    matches = []
    for slide in all_slides:
        # finding the whole region of interest
        min_x = 1e10
        max_x = 0
        min_y = 1e10
        max_y = 0
        n_paths = 0
        for pathologist, slides in annotations.items():
            if slide not in slides:
                continue
            n_paths += 1
            for anno in slides[slide]:
                min_x = min(min_x, min(anno.coords_x))
                max_x = max(max_x, max(anno.coords_x))
                min_y = min(min_y, min(anno.coords_y))
                max_y = max(max_y, max(anno.coords_y))

        annotated_region = np.zeros((max_y - min_y + 1, max_x - min_x + 1, n_paths), dtype='int')

        idp = 0
        for pathologist, slides in annotations.items():
            if slide not in slides:
                continue
            for ida, anno in enumerate(slides[slide]):
                rr, cc = polygon([y - min_y for y in anno.coords_y], [x - min_x for x in anno.coords_x],
                                 (annotated_region.shape[0], annotated_region.shape[1]))
                annotated_region[rr, cc, idp] = ida + 1
            idp += 1

        for idp_ref in range(n_paths - 1):
            ref = annotated_region[..., idp_ref]
            objects = regionprops(ref)

            for obj in tqdm(objects):
                region = annotated_region[max(0, obj.bbox[0] - 10):min(obj.bbox[2] + 10, ref.shape[0]),
                                          max(0, obj.bbox[1] - 10):min(obj.bbox[3] + 10, ref.shape[1])]
                ref_object = region[..., idp_ref] == obj.label
                for idp in range(idp_ref + 1, n_paths):
                    candidates = np.unique(region[ref_object, idp])
                    if candidates.max() == 0:
                        continue  # no match
                    max_iou = 0
                    best_match = 0
                    for c in candidates[candidates > 0]:
                        target_obj = region[..., idp] == c
                        iou = compute_iou(ref_object, target_obj)
                        if iou > max_iou:
                            best_match = c
                            max_iou = iou
                    if best_match > 0:
                        hd = compute_hd(ref_object, region[..., idp] == best_match)
                        matches.append(Match(area=ref_object.sum(),
                                             iou=max_iou,
                                             hd=hd))

    return matches
