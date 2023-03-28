import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops, label, find_contours
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.segmentation import watershed
import openslide
from skimage import draw

from metrics import compute_iou

"""
@author Adrien Foucart

Code for reading and manipulating the MoNuSAC 2020 annotations & team predictions. 

Require the MoNuSAC 2020 testing annotations, available from : https://monusac-2020.grand-challenge.org/Data/
"""

CELL_TYPES = ['Epithelial', 'Lymphocyte', 'Neutrophil', 'Macrophage']
LABELS_CHANNELS = {'Epithelial': 0, 'Lymphocyte': 1, 'Neutrophil': 2, 'Macrophage': 3, 'Ambiguous': 4}
LABELS_COLORS = {
    'Epithelial': np.array([255, 0, 0]),
    'Lymphocyte': np.array([255, 255, 0]),
    'Neutrophil': np.array([0, 0, 255]),
    'Macrophage': np.array([0, 255, 0]),
    'Border': np.array([139, 69, 19])
}


@dataclass
class Nucleus:
    bbox: np.ndarray
    mask: np.ndarray
    area: np.ndarray
    image_path: str
    idx: int

    @property
    def image(self):
        im = imread(f"{self.image_path}.tif")
        return im[self.bbox[0] - 10:self.bbox[2] + 10, self.bbox[1] - 10:self.bbox[3] + 10]


NucleiDict = Dict[str, List[Nucleus]]


def get_all_nuclei(path: str = None) -> NucleiDict:
    """Retrieve all the nuclei in the MoNuSAC annotations, ordered in a dict by cell type."""

    patients = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]

    nuclei = {}
    for cl in CELL_TYPES:
        nuclei[cl] = []

    for pat in patients:
        pat_dir = os.path.join(path, pat)
        images = [f for f in os.listdir(pat_dir) if '_nary' in f]
        for f in tqdm(images):
            nary = np.load(os.path.join(pat_dir, f)).astype('int')
            non_ambiguous = nary[..., 1] != LABELS_CHANNELS['Ambiguous'] + 1
            nary[..., 0] *= non_ambiguous

            props = regionprops(nary[..., 0])

            for obj in props:
                nucleus = Nucleus(bbox=obj.bbox,
                                  mask=np.pad(nary[obj.bbox[0]:obj.bbox[2],
                                              obj.bbox[1]:obj.bbox[3], 0] == obj.label, 10),
                                  area=obj.area,
                                  image_path=os.path.join(pat_dir, f.replace('_nary.npy', '')),
                                  idx=obj.label
                                  )
                cl = CELL_TYPES[nary[nary[..., 0] == obj.label, 1].max() - 1]
                nuclei[cl].append(nucleus)

    return nuclei


def _get_xml_annotations(xml_file: str) -> List[Tuple[str, np.array]]:
    """Reads xml file & returns list of annotations in the form (label_name: str, coords: np.array)"""
    annotations = []
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for attrs, regions, plots in root:
        label_name = attrs[0].attrib['Name']

        for region in regions:
            if region.tag == 'RegionAttributeHeaders':
                continue
            vertices = region[1]
            coords = np.array(
                [[int(float(vertex.attrib['X'])), int(float(vertex.attrib['Y']))] for vertex in vertices]).astype('int')

            annotations.append((label_name, coords))

    return annotations


def _generate_mask(slide: str) -> np.array:
    """Generate n-ary mask from a slide's annotations."""

    wsi = openslide.OpenSlide(f'{slide}.svs')
    size = wsi.level_dimensions[0]
    mask = np.zeros((size[1], size[0], 2)).astype('int')
    annotations = _get_xml_annotations(f'{slide}.xml')

    for idl, (label_name, coords) in enumerate(annotations):
        fill = draw.polygon(coords[:, 1], coords[:, 0], mask.shape)
        mask[fill[0], fill[1], 0] = idl + 1
        mask[fill[0], fill[1], 1] = LABELS_CHANNELS[label_name] + 1

    return mask


def generate_nary_masks_from_annotations(directory: str) -> None:
    """Generate n-ary masks (labeled objects w/ 1 channel per class) from a directory which follows the structure:
    - directory
        - patient folder
            - slide.svs
            - slide.svs
            - ...
        - ...

    Masks are saved as _nary.npy files alongside the .svs file"""
    patients = os.listdir(directory)

    print(f"{len(patients)} patients in directory: {directory}")

    for ip, patient in tqdm(enumerate(patients)):
        patient_dir = os.path.join(directory, patient)
        slides = [f.split('.')[0] for f in os.listdir(patient_dir) if f.split('.')[1] == 'svs']

        for slide in slides:
            mask = _generate_mask(os.path.join(patient_dir, slide))
            np.save(os.path.join(patient_dir, f'{slide}_nary.npy'), mask)

    return


def generate_nary_masks_from_colorcoded(team_dir: str) -> None:
    """Produce n-ary mask from the color-coded images by removing the borders and re-labeling the resulting objects."""
    patients = [p for p in os.listdir(team_dir) if os.path.isdir(os.path.join(team_dir, p))]

    print(f"{len(patients)} patients in directory: {team_dir}")

    for ip, patient in tqdm(enumerate(patients)):
        patient_dir = os.path.join(team_dir, patient)
        files = [f for f in os.listdir(patient_dir) if '_mask.png.tif' in f]
        for f in files:
            cl_im = imread(os.path.join(patient_dir, f))
            bg = cl_im.sum(axis=2) == 0
            borders = (cl_im[..., 0] == LABELS_COLORS["Border"][0]) * \
                      (cl_im[..., 1] == LABELS_COLORS["Border"][1]) * \
                      (cl_im[..., 2] == LABELS_COLORS["Border"][2])
            inner_objects = label((bg ^ borders) == 0)
            full_objects = watershed(edt(borders), markers=inner_objects, mask=(bg == 0))

            nary = np.zeros(cl_im.shape[:2] + (2,))

            for cl, cl_id in LABELS_CHANNELS.items():
                if cl == 'Ambiguous':
                    continue
                inner_class = (cl_im[..., 0] == LABELS_COLORS[cl][0]) * \
                              (cl_im[..., 1] == LABELS_COLORS[cl][1]) * \
                              (cl_im[..., 2] == LABELS_COLORS[cl][2])
                labels = np.unique(full_objects[inner_class])
                for lab in labels:
                    nary[full_objects == lab, 0] = full_objects[full_objects == lab]
                    nary[full_objects == lab, 1] = cl_id + 1
            np.save(os.path.join(patient_dir, f.replace('_mask.png.tif', '_nary.npy')), nary)
    return


def generate_nary_masks_from_teams(directory: str) -> None:
    """Generate the nary masks from the teams colorcoded images"""
    teams = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for team in teams:
        team_dir = os.path.join(directory, team)
        generate_nary_masks_from_colorcoded(team_dir)


def show_nucleus_team_comparison(team_path: str, nuclei: NucleiDict, cl: str, idn: int):
    teams = ["Amirreza Mahbod", "IIAI", "SharifHooshPardaz", "SJTU_426"]
    teams_dir = {}
    for team in teams:
        teams_dir[team] = os.path.join(team_path, team)

    nucleus = nuclei[cl][idn]
    image_file = nucleus.image_path.split('/')[-1]
    im = nucleus.image

    plt.figure(figsize=(15, 5))
    contour_gt = find_contours(nucleus.mask)[0]
    plt.subplot(1, len(teams_dir) + 1, 1)
    plt.imshow(im)
    for idt, (team, team_dir) in enumerate(teams_dir.items()):
        plt.subplot(1, len(teams_dir)+1, idt + 2)
        nary_team = np.load(f'{os.path.join(team_dir, image_file)}_nary.npy')
        region = nary_team[nucleus.bbox[0] - 10:nucleus.bbox[2] + 10, nucleus.bbox[1] - 10:nucleus.bbox[3] + 10]
        possible_matches = np.unique(region[nucleus.mask, 0])
        best_match = 0
        best_iou = 0
        plt.imshow(im)
        plt.plot(contour_gt[:, 1], contour_gt[:, 0], 'b-')

        for match in possible_matches:
            if match == 0:
                continue
            iou = compute_iou(nucleus.mask, region[..., 0] == match)
            if iou > best_iou:
                best_iou = iou
                best_match = match
        if best_match > 0:
            contour_pred = find_contours(region[..., 0] == best_match)[0]
            if CELL_TYPES[int(region[region[..., 0] == best_match, 1].max()) - 1] == cl:
                plt.plot(contour_pred[:, 1], contour_pred[:, 0], 'k--')
                plt.text(contour_pred[0, 1] + 1, contour_pred[0, 0] + 2,
                         CELL_TYPES[int(region[region[..., 0] == best_match, 1].max()) - 1], color='k')
            else:
                plt.plot(contour_pred[:, 1], contour_pred[:, 0], 'w--')
                plt.text(contour_pred[0, 1] + 1, contour_pred[0, 0] + 2,
                         CELL_TYPES[int(region[region[..., 0] == best_match, 1].max()) - 1], color='w')
            plt.title(f'Team {idt + 1}: IoU = {best_iou:.2f}')

        else:
            plt.title(f'Team {idt + 1}: No match')
    plt.show()


class Match:
    """Stores the informations of a matching object pair"""

    def __init__(self, gt_idx, pred_idx, gt_class, pred_class, iou):
        self.gt_idx: int = int(gt_idx)
        self.pred_idx: int = int(pred_idx)
        self.gt_class: int = int(gt_class)
        self.pred_class: int = int(pred_class)
        self.iou: float = iou

    def __str__(self):
        return f"Gt obj {self.gt_idx} (class {self.gt_class}) with " \
               f"pred obj {self.pred_idx} (class {self.pred_class})" \
               f" - IoU = {self.iou:.3f}"


class ImageMatches:
    """Stores the information about all the matches and non-matches in an image."""

    def __init__(self, gt_idxs_class: dict, pred_idxs_class: dict):
        self.gt_idxs_class = gt_idxs_class
        self.pred_idxs_class = pred_idxs_class
        self.matches = []

    def add(self, match: Match):
        self.matches.append(match)


def match_best_iou(gt: np.array, pred: np.array) -> ImageMatches:
    """Find matching pairs of objects between gt & pred mask using the best iou criterion.

    Returns ImageMatches which contains list of matches & list of existing idxs for easy PQ recomputation."""
    nonambiguous_mask = gt[..., 1] != 5
    gt_uid = gt[..., 0] * nonambiguous_mask
    pred_uid = pred[..., 0] * nonambiguous_mask

    gt_idxs = np.unique(gt_uid)
    gt_idxs = gt_idxs[gt_idxs > 0]
    pred_idxs = np.unique(pred_uid)
    pred_idxs = pred_idxs[pred_idxs > 0]

    # prepare quick reference from uid to class
    gt_idxs_class = {}
    for gt_idx in gt_idxs:
        gt_idxs_class[gt_idx] = gt[gt_uid == gt_idx, 1].max()
    pred_idxs_class = {}
    for pred_idx in pred_idxs:
        pred_idxs_class[pred_idx] = pred[pred_uid == pred_idx, 1].max()

    matched_instances = ImageMatches(gt_idxs_class, pred_idxs_class)

    candidate_pairs = []  # first we get all "best matches", then we add them by descending IoU, making sure not to double-match

    # Find matched instances and add it to the list
    for gt_idx in gt_idxs:
        gt_obj_mask = gt_uid == gt_idx
        pred_in_obj = gt_obj_mask * pred_uid

        best_iou = 0
        best_match_idx = 0

        for pred_idx in [idx for idx in np.unique(pred_in_obj) if idx > 0]:
            pred_obj_mask = pred_uid == pred_idx
            intersection = (gt_obj_mask & pred_obj_mask).sum()
            union = (gt_obj_mask | pred_obj_mask).sum()
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_match_idx = pred_idx

        if best_iou > 0:
            candidate_pairs.append([gt_idx, best_match_idx, best_iou])

    candidate_pairs = np.array(candidate_pairs)
    sort_iou = np.argsort(candidate_pairs[:, 2])

    gt_added = []
    pred_added = []

    for gt_idx, pred_idx, iou in candidate_pairs[sort_iou[::-1]]:
        if gt_idx not in gt_added and pred_idx not in pred_added:
            matched_instances.add(Match(gt_idx, pred_idx, gt_idxs_class[gt_idx], pred_idxs_class[pred_idx], iou))
            gt_added.append(gt_idx)
            pred_added.append(pred_idx)

    return matched_instances


def show_matches_under_threshold(image_file: str, idt: int, teams_path: str, annotations_path: str):
    teams = ["Amirreza Mahbod", "IIAI", "SharifHooshPardaz", "SJTU_426"]
    teams_dir = {}
    for team in teams:
        teams_dir[team] = os.path.join(teams_path, team)

    team = teams[idt]
    team_dir = teams_dir[team]
    nary_team = np.load(f'{os.path.join(team_dir, image_file)}_nary.npy')
    nary_gt = np.load(f'{os.path.join(annotations_path, image_file)}_nary.npy')

    matches = match_best_iou(nary_gt, nary_team)

    full_image = imread(f'{os.path.join(annotations_path, image_file)}.tif')

    plt.figure(figsize=(15, 5))
    i = 1
    for match in matches.matches:
        if match.iou < 0.5:
            gt_obj = regionprops((nary_gt[..., 0] == match.gt_idx).astype('int'))[0]
            pred_obj = regionprops((nary_team[..., 0] == match.pred_idx).astype('int'))[0]
            bbox = [
                min(gt_obj.bbox[0], pred_obj.bbox[0]),
                min(gt_obj.bbox[1], pred_obj.bbox[1]),
                max(gt_obj.bbox[2], pred_obj.bbox[2]),
                max(gt_obj.bbox[3], pred_obj.bbox[3])
            ]

            im = full_image[bbox[0] - 5:bbox[2] + 5, bbox[1] - 5:bbox[3] + 5]
            gt_contours = find_contours(nary_gt[..., 0] == match.gt_idx)[0]
            pred_contours = find_contours(nary_team[..., 0] == match.pred_idx)[0]

            plt.subplot(1, 5, i)
            plt.imshow(im)
            plt.plot(gt_contours[:, 1] - bbox[1] + 5, gt_contours[:, 0] - bbox[0] + 5, 'b-')
            plt.plot(pred_contours[:, 1] - bbox[1] + 5, pred_contours[:, 0] - bbox[0] + 5, 'k--')
            plt.title(f'IoU = {match.iou:.3f}')
            i += 1
            if i >= 6:
                break
    plt.show()


def find_image_with_most_nuclei(nuclei: NucleiDict) -> Tuple[int, str]:
    """
    Find the image patch that has the most nuclei in the annotations.
    :param nuclei:
    :return: Tuple with (n_nuclei, path)
    """
    nuclei_per_path = {}

    for cl, nuclei_cl in nuclei.items():
        for nucleus in nuclei_cl:
            if nucleus.image_path not in nuclei_per_path:
                nuclei_per_path[nucleus.image_path] = {}
            if cl not in nuclei_per_path[nucleus.image_path]:
                nuclei_per_path[nucleus.image_path][cl] = 0
            nuclei_per_path[nucleus.image_path][cl] += 1

    max_nuclei = (0, None)
    for path, nuclei_counts in nuclei_per_path.items():
        total = 0
        for cl, count in nuclei_counts.items():
            total += count
        if total > max_nuclei[0]:
            max_nuclei = (total, path)
    return max_nuclei
