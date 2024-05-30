from functools import lru_cache
from tensorflow.keras.utils import Sequence, to_categorical
import numpy
from sklearn.metrics import f1_score, precision_score, recall_score
numpy.float = float
numpy.int = numpy.int_
import random
from itertools import permutations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from PIL import Image
from scipy import ndimage as ndi
from scipy import optimize as op
from sklearn.model_selection import train_test_split
import neuclid as nc
import volume
from neuclid.transformations import superimposition_matrix
from volume import Volume
from scipy import ndimage
from functools import lru_cache
from copy import copy
from tqdm.notebook import tqdm
from functools import lru_cache
import albumentations as aug
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

base_dir= Path("testvolume") # set default path, can replace it with local path


# @jit(target_backend='cuda') # tried to make one functio nworks .....
def load_volume_origin(vol, path):
    vol.load_nrrd(path)
    return vol

class CCS:
    def __init__(self, apex_point, basal_point, rw, side=None):
        self.side = side
        self.apex_point = copy(nc.Point3(apex_point))
        self.origin = copy(nc.Point3(basal_point))
        self.basal_point = self.origin
        self.round_window_center = copy(nc.Point3(rw))
# @jit(target_backend='cuda') # tried to make one functio nworks .....
def transform_volume(
    transform,
    volume,
    rotation_center=None,
    output_shape=None,
    spacing=None,
    order=3,
    fill=0.0,
    only_transform=False,
):

    old_shape = np.array(volume.shape)
    if rotation_center is None:
        rotation_center = volume.to_world(old_shape / 2)  # .mean()
    if spacing is None:
        spacing = np.abs(np.linalg.eigvals(volume.to_world.matrix33)).mean()
    if np.atleast_1d(3).shape == (1,):
        spacing = np.ones(3) * spacing

    if output_shape is None:
        old_extend = np.dot(volume.to_world.matrix33, np.array(volume.shape)).mean()
        output_shape = np.abs((old_extend / spacing)).round().astype("int")
    ijklsource_to_target = nc.Matrix44(transform(volume.to_world))

    target_to_ijksource = ijklsource_to_target.inverse
    to_ijk = nc.Matrix44(np.eye(4))
    to_ijk.matrix33 = np.diag(1 / spacing)
    to_target = to_ijk.inverse
    M = target_to_ijksource(to_target)
    target_center_ijk = np.array(output_shape) / 2
    offset = volume.to_ijk(rotation_center) - np.dot(M.matrix33, target_center_ijk)
    to_target.translation = (np.dot(to_target[:3, :3], -target_center_ijk)) + transform(
        rotation_center
    )  # np.array(i_c.shape)/2))
    M.translation = offset
    if only_transform:
        return to_target
    return Volume(
        ndi.affine_transform(
            volume.data,
            M,
            output_shape=output_shape,
            order=order,
            mode="nearest",
            cval=fill,
        ),
        ijk_to_world=to_target,
        space=volume.space,
    )
def get_coords_from_landmark(filename, space="RAS", return_type="Points"):
    """since  Slicer 4.10_r28794 fcsv files are saved  in 'LPS'
    return_type:  'Points' or 'Pandas'

    """

    try:
        with open(filename, "r") as f:

            for i in range(3):
                l = f.readline()
                if i == 1:
                    fspace_s = l.split("# CoordinateSystem = ")
                    if fspace_s[-1].split("\n")[0] in ["0", "RAS"]:
                        fspace = "RAS"
                    elif fspace_s[-1].split("\n")[0] in ["1", "LPS"]:
                        fspace = "LPS"
                    else:
                        print("space not known")

        data = pd.read_csv(filename, comment="#", header=None)
        if space == "RAS" and fspace == "LPS":
            data.iloc[:, 1:3] *= -1
        elif space == "LPS" and fspace == "RAS":
            data.iloc[:, 1:3] *= -1
        else:
            assert space == fspace

        if return_type == "Points":
            return nc.Points3(data.iloc[:, 1:4].values)
        elif return_type == "Pandas":
            return data
        else:
            print("data type not understiid")
    except:
        # pdb.set_trace()
        return None
def volume_ct_load(base_dir):
    nrrd_volumes = {}
    cochlea_l = {}
    cochlea_r = {}

    cochlea = {"left": cochlea_l, "right": cochlea_r}
    vol = None
    nrrd_files = list(base_dir.glob("*/*.nrrd"))

    unique_parent_names = set()
    unique_paths = []
    for path in nrrd_files:
        parent_name = path.parent.name
        if parent_name not in unique_parent_names:
            unique_parent_names.add(parent_name)
            unique_paths.append(path)

    for path in tqdm(unique_paths):
        print(path.parent.name)
        print(path)
        for side in ["left", "right"]:
            good_segmentations = list(path.parent.glob(side + "/GT*"))
            print(f"\t{side}", end="")
            for seg in good_segmentations:

                nrrd_volumes[path.parent.name[:4]] = path
                print(f"path.parent.name[:4] is {path.parent.name[:4]}")
                print(f"path  is {path}")

                if vol is None:
                    vol = Volume()

                    load_volume_origin(vol, path)
                print(f"\tgood segmentation: {seg.name}")
                cochlea_fcsv_files = list(seg.glob("Cochlea*.fcsv"))
                print(f"cochlea_fcsv_files is {cochlea_fcsv_files}")
                if len(cochlea_fcsv_files) != 1:
                    continue
                    assert (
                        len(cochlea_fcsv_files) == 1
                    ), "More than one or no Cochlea_*.fcsv file found!"
                print(f"\t -> {[f.name for f in cochlea_fcsv_files]}")
                cochlea_center_xyz = get_coords_from_landmark(
                    cochlea_fcsv_files[0], space=vol.space
                )
                del vol
                vol = None
                cochlea[side][path.parent.name[:4]] = cochlea_center_xyz
            print()

    return nrrd_volumes, cochlea_l, cochlea_r



nrrd_volumes, cochlea_l, cochlea_r = volume_ct_load(base_dir)


# print(f"nrrd_volumes is {nrrd_volumes}")

@lru_cache  
def load_volume_ourlabeleddata(d):
    v = Volume()
    v.load_nrrd(nrrd_volumes[d])
    intensity_ = np.percentile(v.data, 99.9)
    intensity_tocompare = np.percentile(v.data, 95)
    if intensity_ > 6 * intensity_tocompare:  # there is an artifact
        mean_intensity = np.mean(v.data)
        std_intensity = np.std(v.data)
        threshold = mean_intensity + 4 * std_intensity
        withoutart = (v.data) < threshold
        print(
            f"hit the artifact! and thresold is {threshold} with the intensity high ={intensity_}"
        )
        v.data[withoutart[:, :, :] == False] = 0

    d_min = np.min(v.data)
    d_max = np.max(v.data)

    def normalizer(x):
        return (x - d_min) / (d_max - d_min)

    v.data = normalizer(v.data)
    v.data = v.data.astype("float32")

    return v, normalizer


def load_volume_ijk_GT(volume_indice):
    v_0012, normaliser = load_volume_ourlabeleddata(d=volume_indice)

    c_l1 = CCS(*cochlea_l[volume_indice], side="left")
    cochlea_center_l = c_l1.origin
    cochlea_center_l_apex = c_l1.apex_point
    cochlea_center_l_rw = c_l1.round_window_center

    c_r1 = CCS(*cochlea_r[volume_indice], side="right")
    cochlea_center_r = c_r1.origin
    cochlea_center_r_apex = c_r1.apex_point
    cochlea_center_r_rw = c_r1.round_window_center

    right_ijk_0012 = v_0012.to_ijk(cochlea_center_r)
    left_ijk_0012 = v_0012.to_ijk(cochlea_center_l)
    right_ijk_0012_apex = v_0012.to_ijk(cochlea_center_r_apex)
    left_ijk_0012_apex = v_0012.to_ijk(cochlea_center_l_apex)
    right_ijk_0012_rw = v_0012.to_ijk(cochlea_center_r_rw)
    left_ijk_0012_rw = v_0012.to_ijk(cochlea_center_l_rw)

    return (
        v_0012,
        cochlea_center_l,
        cochlea_center_l_apex,
        cochlea_center_l_rw,
        left_ijk_0012,
        left_ijk_0012_apex,
        left_ijk_0012_rw,
        cochlea_center_r,
        cochlea_center_r_apex,
        cochlea_center_r_rw,
        right_ijk_0012,
        right_ijk_0012_apex,
        right_ijk_0012_rw,
    )

def crop_manually_withoutaxial(v, landmark_ijk, window_size=81):

    half_size = window_size // 2
    i_1, j_1, k_1 = np.array(landmark_ijk)

    i_1, j_1, k_1 = round(i_1), round(j_1), round(k_1)

    if not (
        half_size <= np.minimum(v.shape[0] - i_1, i_1)
        and half_size <= np.minimum(v.shape[1] - j_1, j_1)
        and half_size <= np.minimum(v.shape[2] - k_1, k_1)
    ):
        print(
            "volume ct too small to crop this big windowsize, consider smaller window size!! "
        )
        if not (half_size <= np.minimum(v.shape[0] - i_1, i_1)):
            half_size_small_i = np.minimum(v.shape[0] - i_1, i_1) - 1

        else:
            half_size_small_i = half_size
        if not half_size <= np.minimum(v.shape[1] - j_1, j_1):
            half_size_small_j = np.minimum(v.shape[1] - j_1, j_1) - 1

        else:
            half_size_small_j = half_size
        if not half_size <= np.minimum(v.shape[2] - k_1, k_1):

            half_size_small_k = np.minimum(v.shape[2] - k_1, k_1) - 1
        else:
            half_size_small_k = half_size

        i_padding = max(0, window_size - int(2 * half_size_small_i + 1))
        j_padding = max(0, window_size - int(2 * half_size_small_j + 1))
        k_padding = max(0, window_size - int(2 * half_size_small_k + 1))

        crop_leftsagittal_small = v[
            i_1,
            j_1 - half_size_small_j - 1 : j_1 + half_size_small_j,
            k_1 - half_size_small_k - 1 : k_1 + half_size_small_k,
        ]
        crop_leftsagittal_small = np.squeeze(crop_leftsagittal_small.data, axis=0)

        crop_leftsagittal = np.pad(
            crop_leftsagittal_small,
            ((0, j_padding), (0, k_padding)),
            mode="constant",
            constant_values=np.min(crop_leftsagittal_small),
        )

        crop_leftcoronal_small = v[
            i_1 - half_size_small_i - 1 : i_1 + half_size_small_i,
            j_1,
            k_1 - half_size_small_k - 1 : k_1 + half_size_small_k,
        ]
        crop_leftcoronal_small = np.squeeze(crop_leftcoronal_small.data, axis=1)

        crop_leftcoronal = np.pad(
            crop_leftcoronal_small,
            ((0, i_padding), (0, k_padding)),
            mode="constant",
            constant_values=np.min(crop_leftcoronal_small),
        )

    else:

        crop_leftsagittal = v[
            i_1,
            j_1 - half_size - 1 : j_1 + half_size,
            k_1 - half_size - 1 : k_1 + half_size,
        ]
        crop_leftcoronal = v[
            i_1 - half_size - 1 : i_1 + half_size,
            j_1,
            k_1 - half_size - 1 : k_1 + half_size,
        ]
        crop_leftsagittal = np.squeeze(crop_leftsagittal.data, axis=0)

        crop_leftcoronal = np.squeeze(crop_leftcoronal.data, axis=1)

    return crop_leftsagittal, crop_leftcoronal


def bothside_backpoints(
    v,
    landmark_cochela_basal_left,
    landmark_cochela_apex_left,
    landmark_cochela_rw_left,
    landmark_cochela_basal_right,
    landmark_cochela_apex_right,
    landmark_cochela_rw_right,
    threshold=1,
    collect_range=41,
):

    i_1, j_1, k_1 = np.array(landmark_cochela_basal_left)

    print(f"bothside_backpoints: collect_range is {collect_range}")

    i_1, j_1, k_1 = round(i_1), round(j_1), round(k_1)
    # apex
    i_2, j_2, k_2 = np.array(landmark_cochela_apex_left)

    i_2, j_2, k_2 = round(i_2), round(j_2), round(k_2)

    # rw
    i_3, j_3, k_3 = np.array(landmark_cochela_rw_left)

    i_3, j_3, k_3 = round(i_3), round(j_3), round(k_3)

    collect_range = collect_range
    list_boader = (
        v.shape[0] - i_1,
        i_1,
        v.shape[1] - j_1,
        j_1,
        v.shape[2] - k_1,
        k_1,
    )

    if collect_range > min(list_boader):
        collect_range = min(list_boader)
    selected_backgroundpoints = []

    for x in range(i_1 - collect_range, i_1 + collect_range - 1):
        for y in range(j_1 - collect_range, j_1 + collect_range - 1):
            for z in range(k_1 - collect_range, k_1 + collect_range - 1):
                if (
                    (x > i_1 + threshold or x < i_1 - threshold)
                    and (y > j_1 + threshold or y < j_1 - threshold)
                    and (z > k_1 + threshold or z < k_1 - threshold)
                    and (x > i_2 + threshold or x < i_2 - threshold)
                    and (y > j_2 + threshold or y < j_2 - threshold)
                    and (z > k_2 + threshold or z < k_2 - threshold)
                    and (x > i_3 + threshold or x < i_3 - threshold)
                    and (y > j_3 + threshold or y < j_3 - threshold)
                    and (z > k_3 + threshold or z < k_3 - threshold)
                ):
                    selected_backgroundpoints.append(np.array([x, y, z]))

    "right side cochleaclass points"
    # basal to eviter
    i_4, j_4, k_4 = np.array(landmark_cochela_basal_right)

    i_4, j_4, k_4 = round(i_4), round(j_4), round(k_4)
    # apex
    i_5, j_5, k_5 = np.array(landmark_cochela_apex_right)

    i_5, j_5, k_5 = round(i_5), round(j_5), round(k_5)

    # rw
    i_6, j_6, k_6 = np.array(landmark_cochela_rw_right)

    i_6, j_6, k_6 = round(i_6), round(j_6), round(k_6)

    # Set the range for collecting points
    collect_range2 = collect_range
    list_boader2 = (
        v.shape[0] - i_4,
        i_4,
        v.shape[1] - j_4,
        j_4,
        v.shape[2] - k_4,
        k_4,
    )

    if collect_range2 > min(list_boader2):
        collect_range2 = min(list_boader2)

    for x2 in range(i_4 - collect_range2, i_4 + collect_range2 - 1):
        for y2 in range(j_4 - collect_range2, j_4 + collect_range2 - 1):
            for z2 in range(k_4 - collect_range2, k_4 + collect_range2 - 1):
                if (
                    (x2 > i_4 + threshold or x2 < i_4 - threshold)
                    and (y2 > j_4 + threshold or y2 < j_4 - threshold)
                    and (z2 > k_4 + threshold or z2 < k_4 - threshold)
                    and (x2 > i_5 + threshold or x2 < i_5 - threshold)
                    and (y2 > j_5 + threshold or y2 < j_5 - threshold)
                    and (z2 > k_5 + threshold or z2 < k_5 - threshold)
                    and (x2 > i_6 + threshold or x2 < i_6 - threshold)
                    and (y2 > j_6 + threshold or y2 < j_6 - threshold)
                    and (z2 > k_6 + threshold or z2 < k_6 - threshold)
                ):
                    selected_backgroundpoints.append(np.array([x2, y2, z2]))
    return selected_backgroundpoints


def tfimage(img):
    img = img.reshape(*img.shape, 1)
    window = tf.keras.preprocessing.image.array_to_img(img)
    window = tf.keras.preprocessing.image.img_to_array(window)
    # window=tf.expand_dims(window, axis=0) # only use it for single image testing can try to use tf for expand :candidate_slice=tf.expand_dims(candidate_slice,axis=2)
    window_input = (
        window / 255.0
    )  # can't remove it!!! even after v normalised, if remove, the image is not in range 0 and 1
    return window_input


class LimitSpatialTransforms(ImageOnlyTransform):
    def __init__(self, transform, center_size=(30, 30), always_apply=False, p=1.0):
        super(LimitSpatialTransforms, self).__init__(always_apply, p)

        self.transform = transform
        self.center_size = center_size

    def apply(self, img, **params):
        h, w = img.shape[:2]

        ch, cw = self.center_size

        top = (h - ch) // 2
        left = (w - cw) // 2
        bottom = top + ch
        right = left + cw

        img_center = img[top:bottom, left:right]
        img_transformed = self.transform(image=img, **params)["image"]
        img_transformed[top:bottom, left:right] = img_center

        return img_transformed



augmentation_all = aug.OneOf(
    [
        aug.GaussNoise(
            var_limit=(0, 0.0006), mean=0.0002, p=1
        ),  ## p here is the probabilty to be applied inside this one of
        aug.GaussianBlur(
            blur_limit=(1, 3), p=1
        ),  # blur kernel 1 bis 7 pixels, bigger, blurer
        aug.Sharpen(
            alpha=(0.002, 0.007), lightness=(0.1, 0.4), p=1
        ),  # alpha bigger -> stronger sharpening
    ],
    p=1,
)

class ImageIterator:
    def __init__(self, image):
        self.image = image

    def __iter__(self):
        return self

    def __next__(self):
        return self.image
    

class custom_gradual_padding_class:
    def __init__(self, center_size_custom_gradual_padding=71):
        self.center_size_custom_gradual_padding = center_size_custom_gradual_padding
        self.center_size_custom_gradual_padding_reset = 71

    def custom_gradual_padding(self, image, crop_size=81):

        print(
            f" custom_gradual_padding:  is center_size {self.center_size_custom_gradual_padding} "
        )  # 4

        h, w = image.shape[:2]  # 81x81

        center_h = round((h - self.center_size_custom_gradual_padding) / 2)

        center_w = round((w - self.center_size_custom_gradual_padding) / 2)

        cropped = image[
            center_h : center_h + self.center_size_custom_gradual_padding,
            center_w : center_w + self.center_size_custom_gradual_padding,
        ]

        pad_top = (crop_size - self.center_size_custom_gradual_padding) // 2

        pad_bottom = crop_size - self.center_size_custom_gradual_padding - pad_top

        pad_left = (crop_size - self.center_size_custom_gradual_padding) // 2

        pad_right = crop_size - self.center_size_custom_gradual_padding - pad_left


        padded = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

        self.center_size_custom_gradual_padding += 2
        if self.center_size_custom_gradual_padding > crop_size:
            print(f" hit ")  # 4

            self.center_size_custom_gradual_padding = (
                self.center_size_custom_gradual_padding_reset
            )

        return padded
    

def image_volume_rotator_distriutionajusted_volume(
    v,
    center,
    alpha_range,
    beta_range,
    gamma_range,
    steps1,
    steps2,
    steps3,
    rotated_shape,
    order,
):
    alpha_list_uniform = []
    alpha_list_choice = []
    beta_list_uniform = []
    beta_list_choice = []
    distribution = 0.2
    alpha_range, beta_range, gamma_range = (
        np.deg2rad(alpha_range),
        np.deg2rad(beta_range),
        np.deg2rad(gamma_range),
    )

    for alpha in np.random.uniform(low=0, high=alpha_range, size=(steps1,)):

        alpha_list_uniform.append(alpha)
    for i in range(len(alpha_list_uniform)):

        initial_p_alpha = (1 - distribution) / len(alpha_list_uniform)
        list_p_alpha = [initial_p_alpha] * (len(alpha_list_uniform))
        where_0_alpha = alpha_list_uniform.index(min(alpha_list_uniform))

        list_p_alpha[where_0_alpha] = list_p_alpha[where_0_alpha] + distribution
        alpha_sortie = np.random.choice(alpha_list_uniform, p=list_p_alpha)
        alpha_sortie = alpha_sortie * np.random.uniform(low=0.8, high=1.1)

        alpha_list_choice.append(alpha_sortie)

    if steps1 >= 3:
        a = round(steps1 * 0.2)
        alpha_list_choice[: 2 * a] = [0, 0] * a
    else:
        raise ValueError("steps1 is too little!")

    for beta in np.random.uniform(low=0, high=beta_range, size=(steps2,)):

        beta_list_uniform.append(beta)

    for j in range(len(beta_list_uniform)):
        initial_p_beta = (1 - distribution) / len(beta_list_uniform)
        list_p_beta = [initial_p_beta] * (len(beta_list_uniform))
        where_0_beta = beta_list_uniform.index(min(beta_list_uniform))

        list_p_beta[where_0_beta] = list_p_beta[where_0_beta] + distribution
        beta_sortie = np.random.choice(beta_list_uniform, p=list_p_beta)
        beta_sortie = beta_sortie * np.random.uniform(low=0.8, high=1.1)

        beta_list_choice.append(beta_sortie)

    if steps2 >= 3:
        b = round(steps2 * 0.2)
        beta_list_choice[: 2 * b] = [0, 0] * b
    else:
        raise ValueError("steps2 is too little!")

    t_translation = nc.Matrix44.create_from_translation(center)
    t_translation_inverse = t_translation.inverse
    for alpha in alpha_list_choice:
        for beta in beta_list_choice:
            for gamma in np.random.uniform(low=0, high=gamma_range, size=(steps3,)):
                t_axial = nc.Matrix44(
                    nc.transformations.euler_matrix(alpha, beta, gamma)
                )
                t_sagittal = nc.Matrix44(
                    nc.transformations.euler_matrix(gamma, beta, alpha)
                )
                t_coron = nc.Matrix44(
                    nc.transformations.euler_matrix(alpha, gamma, beta)
                )

                if rotated_shape[2] == 1:
                    t = t_axial
                elif rotated_shape[0] == 1:
                    t = t_sagittal

                elif rotated_shape[1] == 1:
                    t = t_coron

                else:
                    t = t_axial
                rotated_volume = volume.transform_volume(
                    t_translation(t(t_translation_inverse)),
                    v,
                    rotation_center=nc.Point3(center),
                    output_shape=rotated_shape,
                    order=order,
                )
                yield rotated_volume

def cochleaclass_generator_withrotation_withoutaxial(
    volume_name,
    landmark_ijk,
    cochlea_world_fiducials,
    id_indextotal,
    crop_manually_cache,
    crop_smallvolume_cache,
    image_volume_rotator_distriutionajusted_volume_cache,
    single_classlen,
    alpha_range,
    beta_range,
    gamma_range,
    steps1,
    steps2,
    steps3,
    window_size,
    paddinglist_clear,
    padding_crop_axial_list,
    padding_crop_coronal_list,
    padding_crop_sagittal_list,
    gradual_rotated_padding_size,
    rotated_augmentation_size,
    indice,
    order,
    justforiterationindex,
):

    i_1, j_1, k_1 = np.array(landmark_ijk)

    i_1, j_1, k_1 = round(i_1), round(j_1), round(k_1)

    key = (volume_name, str(landmark_ijk))

    if key not in crop_manually_cache:
        """1.1 run manual origin crop =slice, the core cohclea 3 direction"""
        crop_manually_cache[key] = []
        crop_manually_cache[key].append(
            crop_manually_withoutaxial(volume_name, landmark_ijk, window_size=81)
        )
        crop_manually_cache[key].append(custom_gradual_padding_class())

    if key not in crop_smallvolume_cache:
        """for 1.5 rotation: the whole sub volume cropped"""

        window_crop = round(window_size * 1.6)
        crop_smallvolume_cache[key] = volume_name[
            int(i_1 - window_crop / 2) : int(i_1 + window_crop / 2),
            int(j_1 - window_crop / 2) : int(j_1 + window_crop / 2),
            int(k_1 - window_crop / 2) : int(k_1 + window_crop / 2),
        ]

    crop_sagittal = crop_manually_cache[key][0][0]

    crop_coronal = crop_manually_cache[key][0][1]
    center_size = 71
    padding_size = round((window_size - center_size) / 2)  # 24

    custom_gradual_padding_eachclass = crop_manually_cache[key][1]

    v_data_croparoundcochlea = crop_smallvolume_cache[key]

    if key not in image_volume_rotator_distriutionajusted_volume_cache:
        image_volume_rotator_distriutionajusted_volume_cache[key] = []
        # sagittal
        image_volume_rotator_distriutionajusted_volume_cache[key].append(
            image_volume_rotator_distriutionajusted_volume(
                v_data_croparoundcochlea,
                cochlea_world_fiducials,
                alpha_range=alpha_range,
                beta_range=beta_range,
                gamma_range=gamma_range,
                steps1=steps1,
                steps2=steps2,
                steps3=steps3,
                rotated_shape=(1, window_size, window_size),
                order=order,
            )
        )
        # coronal
        image_volume_rotator_distriutionajusted_volume_cache[key].append(
            image_volume_rotator_distriutionajusted_volume(
                v_data_croparoundcochlea,
                cochlea_world_fiducials,
                alpha_range=alpha_range,
                beta_range=beta_range,
                gamma_range=gamma_range,
                steps1=steps1,
                steps2=steps2,
                steps3=steps3,
                rotated_shape=(window_size, 1, window_size),
                order=order,
            )
        )

    v_rotated_generator_sagittal = image_volume_rotator_distriutionajusted_volume_cache[
        key
    ][0]
    v_rotated_generator_coronal = image_volume_rotator_distriutionajusted_volume_cache[
        key
    ][1]

    if id_indextotal <= 1:

        if id_indextotal == 0:
            cochlea_1_generator = ImageIterator(crop_sagittal)
        elif id_indextotal == 1:
            cochlea_1_generator = ImageIterator(crop_coronal)

        """1.2  augmentation on original cochlea 2 direction,  size can be variated or can change"""
    elif (
        1
        < id_indextotal
        <= round((single_classlen + justforiterationindex - 2) / 9 * 0.2)
    ):
        # 10% to do the augmeantion

        rand_num = random.random()
        if rand_num <= 0.55:
            cochlea_1_generator = ImageIterator(
                augmentation_all(image=crop_sagittal)["image"]
            )
        else:
            cochlea_1_generator = ImageIterator(
                augmentation_all(image=crop_coronal)["image"]
            )
    elif (
        round((single_classlen + justforiterationindex - 2) / 9 * 0.2)
        < id_indextotal
        <= round((single_classlen + justforiterationindex - 2) / 9 * 0.3)
    ):

        rand_num = random.random()

        if id_indextotal <= 2 * padding_size + round(
            (single_classlen + justforiterationindex - 2) / 9 * 0.2
        ):
            if id_indextotal <= padding_size + round(
                (single_classlen + justforiterationindex - 2) / 9 * 0.2
            ):
                padded = custom_gradual_padding_eachclass.custom_gradual_padding(
                    crop_sagittal
                )
                cochlea_1_generator = ImageIterator(padded)

                padding_crop_sagittal_list.append(padded)

            else:
                center_size_custom_gradual_padding_cornoal = random.randint(71, 79)
                custom_gradual_padding_eachclass = custom_gradual_padding_class(
                    center_size_custom_gradual_padding_cornoal
                )

                padded = custom_gradual_padding_eachclass.custom_gradual_padding(
                    crop_coronal
                )
                cochlea_1_generator = ImageIterator(padded)
                padding_crop_coronal_list.append(padded)

            """1.4 2d augmenation on every padding image -> size can be variated or can change (single_classlen-3)* 0.1"""

        else:  # the rest all do the augmenation of the padding,

            if rand_num <= 0.45:  # 0.45 + 0.30

                rand_pad = len(padding_crop_sagittal_list) - 1
                if rand_pad <= 1:
                    rand_pad = 1
                random_value_pad = random.randint(0, rand_pad)

                crop_sagittal = padding_crop_sagittal_list[random_value_pad]

                cochlea_1_generator = ImageIterator(
                    augmentation_all(image=crop_sagittal)["image"]
                )
            else:

                rand_pad_ = len(padding_crop_coronal_list) - 1
                if rand_pad_ <= 1:
                    rand_pad_ = 1
                random_value_pad = random.randint(0, rand_pad_)

                crop_coronal = padding_crop_coronal_list[random_value_pad]

                cochlea_1_generator = ImageIterator(
                    augmentation_all(image=crop_coronal)["image"]
                )

        """1.5 rotation: start from a half a single class: 
                   35% on axial direction rotation + padding on rotated + augmentation on rota
                """
    elif id_indextotal > round((single_classlen + justforiterationindex - 2) / 9 * 0.3):

        rand_num = random.random()

        if rand_num <= 0.5:

            """rotate only take one sagittal slice"""
            cropped_rotated_vol_slice1 = next(v_rotated_generator_sagittal)

            cochlea_1 = np.squeeze(cropped_rotated_vol_slice1.data, axis=0)
            cochlea_1_generator = ImageIterator(cochlea_1)

            if id_indextotal % gradual_rotated_padding_size == 0:
                center_size_custom_gradual_padding_rotated_sagittal = random.randint(
                    71, 79
                )
                custom_gradual_padding_eachclass = custom_gradual_padding_class(
                    center_size_custom_gradual_padding_rotated_sagittal
                )
                cochlea_1_rotated = (
                    custom_gradual_padding_eachclass.custom_gradual_padding(cochlea_1)
                )
                cochlea_1_generator = ImageIterator(cochlea_1_rotated)

            elif id_indextotal % rotated_augmentation_size == 0:
                cochlea_1_generator = ImageIterator(
                    augmentation_all(image=cochlea_1)["image"]
                )

        else:
            cropped_rotated_vol_slice1 = next(v_rotated_generator_coronal)
            cochlea_1 = np.squeeze(cropped_rotated_vol_slice1.data, axis=1)
            cochlea_1_generator = ImageIterator(cochlea_1)

            if id_indextotal % gradual_rotated_padding_size == 0:
                center_size_custom_gradual_padding_rotated_coronal = random.randint(
                    71, 79
                )
                custom_gradual_padding_eachclass = custom_gradual_padding_class(
                    center_size_custom_gradual_padding_rotated_coronal
                )
                cochlea_1_rotated = (
                    custom_gradual_padding_eachclass.custom_gradual_padding(cochlea_1)
                )
                cochlea_1_generator = ImageIterator(cochlea_1_rotated)

            elif id_indextotal % rotated_augmentation_size == 0:
                cochlea_1_generator = ImageIterator(
                    augmentation_all(image=cochlea_1)["image"]
                )

    return cochlea_1_generator


def backgroundclass_generator_little(
    volume_name,
    left_ijk_origin,
    left_ijk_apex,
    left_ijk_rw,
    right_ijk_origin,
    right_ijk_apex,
    right_ijk_rw,
    # total_nb,
    selected_backgroundpoints,
    id_indextotal,
    padding_augmentationsize=45,
):

    min_val = min(id_indextotal, (len(selected_backgroundpoints) - 1))
    random_value_back = random.randint(min_val, (len(selected_backgroundpoints) - 1))

    landmark_ijk_background = selected_backgroundpoints[random_value_back]
    assert not (
        np.array_equal(landmark_ijk_background.round(), left_ijk_origin.round())
        or (np.array_equal(landmark_ijk_background.round(), left_ijk_apex.round()))
        or (np.array_equal(landmark_ijk_background.round(), left_ijk_rw.round()))
        or (np.array_equal(landmark_ijk_background.round(), right_ijk_origin.round()))
        or (np.array_equal(landmark_ijk_background.round(), right_ijk_apex.round()))
        or (np.array_equal(landmark_ijk_background.round(), right_ijk_rw.round()))
    ), "backgroundclass_generator_little: background image touch the cochlea label!!!"
    cropbackground_sagittal, cropbackground_coronal = crop_manually_withoutaxial(
        volume_name, landmark_ijk_background, window_size=81
    )

    center_size_custom_gradual_padding_back = random.randint(71, 79)
    custom_gradual_padding_eachclass = custom_gradual_padding_class(
        center_size_custom_gradual_padding_back
    )
    rand_num = random.random()
    if rand_num <= 0.9:

        if id_indextotal % 2 == 0:
            background_generator = ImageIterator(cropbackground_sagittal)
        else:
            background_generator = ImageIterator(cropbackground_coronal)

        """1.2  augmentation on each of the core cohclea 3 direction,  size can be variated or can change"""

    elif rand_num <= 0.95:  # do padding

        if id_indextotal % 2 == 0:  # saggital
            padded = custom_gradual_padding_eachclass.custom_gradual_padding(
                cropbackground_sagittal
            )
            background_generator = ImageIterator(padded)

        else:

            padded = custom_gradual_padding_eachclass.custom_gradual_padding(
                cropbackground_coronal
            )
            background_generator = ImageIterator(padded)

        """1.4 2d augmenation on every padding image -> size can be variated or can change (single_classlen-3)* 0.1"""

    else:
        if id_indextotal % 2 == 0:

            background_generator = ImageIterator(
                augmentation_all(image=cropbackground_sagittal)["image"]
            )
        else:

            background_generator = ImageIterator(
                augmentation_all(image=cropbackground_coronal)["image"]
            )
    return background_generator


class Data_augmentation_Generator(
    Sequence
):

    """
    ###
    changed the background points every epoch: make it increment every epoch not just repeat the same indice
    one epoch has 300 id_indextotal all 3 classes
    then epoch2, start at id_indextotal = 201(only background)

    -----------4 classes----
    class_matrix[0]: cochlea apex
    class_matrix[1]: cochlea basal
    class_matrix[2]: cochlea RW
    class_matrix[3]: background

    """

    integer_labels = [0, 1, 2, 3]  # 4, 5, 6] # to remian background rw and basal f
    num_classes = 4

    def __init__(
        self,
        batch_size=32,
        total_epoch=120,
        # cochlea_l=None,
        # cochlea_r=None,
        nrrd_volumes=nrrd_volumes,
        selected_volumes=None,
        steps1=None,
        steps2=None,
        steps3=None,
        alpha_range=None,
        beta_range=None,
        gamma_range=None,
        window_size=None,  # 81
        gradual_rotated_padding_size=None,
        rotated_augmentation_size=None,
    ):
        "Initialization"
        assert nrrd_volumes is not None, "nrrd_volumes have to be provided"
        self.class_matrix = to_categorical(
            Data_augmentation_Generator.integer_labels,
            Data_augmentation_Generator.num_classes,
        )
        # self.cochlea_l=cochlea_l,
        # self.cochlea_r=cochlea_r,
        self.nrrd_volumes = nrrd_volumes
        self.batch_size = batch_size
        self.steps1 = steps1
        self.steps2 = steps2
        self.steps3 = steps3

        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range

        self.window_size = window_size  # 81
        self.gradual_rotated_padding_size = gradual_rotated_padding_size
        self.rotated_augmentation_size = rotated_augmentation_size
        self.total_epoch = total_epoch

        self.id_indextotal = 0
        self.id_indextotal_leftapex = 0
        self.id_indextotal_rightapex = 0
        self.id_indextotal_leftorigin = 0
        self.id_indextotal_rightorigin = 0
        self.id_indextotal_leftrw = 0
        self.id_indextotal_rightrw = 0
        self.id_indextotal_background = 0

        self.crop_manually_cache = {}
        self.crop_smallvolume_cache = {}
        self.image_volume_rotator_distriutionajusted_volume_cache = {}
        self.volume_class_indexlist = list()
        self.volume_class_indexlist_background = list()
        self.padding_crop_axial_list = list()
        self.padding_crop_coronal_list = list()
        self.padding_crop_sagittal_list = list()
        self.indice = 0
        self.tempo_id_indextotal = 0
        self.first_indexadjustment = False
        self.current_index = -1
        self.klass = self.class_matrix[0]
        self.previous_klass = None
        self.selected_backgroundpoints = None
        self.switch_volumecount = 0
        self.rotater_dicreset = False
        self.epoch_count = 0
        self.left_right_index = 0
        self.num_classes_distribution = 9

        if selected_volumes is None:
            self.selected_volumes = self.nrrd_volumes.keys()
        else:
            self.selected_volumes = selected_volumes

    def __len__(self):
        N = 0

        n = (
            len(self.selected_volumes)
            * (
                Data_augmentation_Generator.num_classes
                + 5
                - self.tempo_id_indextotal
            )
            * (self.steps1 * self.steps2 * self.steps3)
            - 1
        )

        N += n
        return N // self.batch_size

    def __getitem__(self, index):
        "Generate one batch of data"
        self.current_index += 1
        index = self.current_index % self.__len__()  # divid by N
        start_index = index * self.batch_size
        end_index = min(
            (index + 1) * self.batch_size,
            len(self.selected_volumes)
            * (
                Data_augmentation_Generator.num_classes
                + 5
                - self.tempo_id_indextotal
            )
            * self.steps1
            * self.steps2
            * self.steps3,
        )

        # Generate data
        X, y = zip(
            *[next(self.gen_input_data()) for i in range(start_index, end_index)]
        )

        # Convert the output to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # print(f"__getitem__index X shape is{X.shape}")

        return X, y

    def gen_input_data(
        self,
    ):

        single_classlen = self.steps1 * self.steps2 * self.steps3
        paddinglist_clear = True

        justforiterationindex = 0

        print(
            f"gen_input_data in the very begining !! self.id_indextotal  is  {self.id_indextotal }"
        )

        """
        here to switch volumes based on id_indextotal, but id_indextotal is always bigger than 实际长度(one batch size)
            start from 2nd volume, can't ajouter more than 实际长度
        """

        if self.id_indextotal >= (single_classlen + justforiterationindex) * (
            Data_augmentation_Generator.num_classes
            + 5
            - self.tempo_id_indextotal
        ):
            print(
                f" gen_input_data hit another volume start from the {self.indice} volume!!! !!!!!!!!!!"
            )
            self.indice += 1
            self.id_indextotal = 0
            self.id_indextotal_leftapex = 0
            self.id_indextotal_rightapex = 0
            self.id_indextotal_leftorigin = 0
            self.id_indextotal_rightorigin = 0
            self.id_indextotal_leftrw = 0
            self.id_indextotal_rightrw = 0
            self.switch_volumecount += 1
        self.rotater_dicreset = False
        if self.switch_volumecount > (len(self.selected_volumes) - 1):
            self.indice = 0
            self.rotater_dicreset = True
            self.switch_volumecount = 0
            self.volume_class_indexlist = list()
            self.epoch_count += 1
            self.left_right_index = 0

            print(f"gen_input_data: go to next epoch {self.epoch_count}!!")
        if self.rotater_dicreset == True:
            self.crop_manually_cache = {}
            self.crop_smallvolume_cache = {}
            self.image_volume_rotator_distriutionajusted_volume_cache = {}

        volume_indice = self.selected_volumes[self.indice]
        
        # print(f"gen_input_data: selected_volumes is {self.selected_volumes} !!!!!!!!!!")

        # assert (
        #     volume_indice in self.cochlea_l[0] and volume_indice in self.cochlea_r[0]
        # ), f"gen_input_data: volume_indice is {volume_indice} and self.cochlea_l[0] is {self.cochlea_l[0]} and self.cochlea_r.keys() is {self.cochlea_r}, and nrrd_volumes is {self.nrrd_volumes}This volume don't have both side ear fiducial lists!!"
        
        # print(f"gen_input_data: volume_indice is {volume_indice} and self.cochlea_l[0] is {self.cochlea_l[0]} and self.cochlea_r.keys() is {self.cochlea_r}, and nrrd_volumes is {self.nrrd_volumes}")

        # """used to not always generate new padding_gradual every itertation for the same volume"""
        volume_name = "v_" + str(volume_indice)
        print(f"gen_input_data:volume_nameis {volume_name} !!!!!!!!!!")

        if volume_name not in self.volume_class_indexlist:
            self.volume_class_indexlist.append(volume_name)
            self.id_indextotal = 0  #
            self.id_indextotal_leftapex = 0
            self.id_indextotal_rightapex = 0
            self.id_indextotal_leftorigin = 0
            self.id_indextotal_rightorigin = 0
            self.id_indextotal_leftrw = 0
            self.id_indextotal_rightrw = 0
            self.left_right_index = 0

            # print("gen_input_data:self.volume_indexlist ajouter!")

            (
                volume_name,
                cochlea_center_l_origin,
                cochlea_center_l_apex,
                cochlea_center_l_rw,
                left_ijk_origin,
                left_ijk_apex,
                left_ijk_rw,
                cochlea_center_r_origin,
                cochlea_center_r_apex,
                cochlea_center_r_rw,
                right_ijk_origin,
                right_ijk_apex,
                right_ijk_rw,
            ) = load_volume_ijk_GT(volume_indice=volume_indice)
            self.selected_backgroundpoints = bothside_backpoints(
                volume_name,
                left_ijk_origin,
                left_ijk_apex,
                left_ijk_rw,
                right_ijk_origin,
                right_ijk_apex,
                right_ijk_rw,
                threshold=2,
                collect_range=38,
            )
            self.id_indextotal_background = 0
        else:
            (
                volume_name,
                cochlea_center_l_origin,
                cochlea_center_l_apex,
                cochlea_center_l_rw,
                left_ijk_origin,
                left_ijk_apex,
                left_ijk_rw,
                cochlea_center_r_origin,
                cochlea_center_r_apex,
                cochlea_center_r_rw,
                right_ijk_origin,
                right_ijk_apex,
                right_ijk_rw,
        ) = load_volume_ijk_GT(volume_indice=volume_indice)
        class_switch = self.id_indextotal // single_classlen
        maxclass_indice = (
            Data_augmentation_Generator.num_classes
            - 1
        )  # here to change when change the nb of classes
        if class_switch > maxclass_indice:
            class_switch = maxclass_indice
        self.klass = self.class_matrix[class_switch]

        if self.previous_klass is not None and not np.array_equal(
            self.previous_klass, self.klass
        ):
            paddinglist_clear = False
        if paddinglist_clear == False:
            self.padding_crop_axial_list = list()
            self.padding_crop_coronal_list = list()
            self.padding_crop_sagittal_list = list()

        """0. order switch"""
        order = 3
        # if c<20:
        # if self.id_indextotal % 2 == 0:
        #     order = 3
        # else:
        #     order = 1
        #     print("order switch")
        if self.id_indextotal > 0 and (self.id_indextotal % 9 == 0):
            self.num_classes_distribution = random.randint(6, 9)
        class_id = self.id_indextotal % self.num_classes_distribution
        if class_id > maxclass_indice:
            class_id = maxclass_indice
        """##########class 1: cochlea apex  #####"""
        print("gen_input_data: start class 1 function1!")
        if class_id == 0:

            if self.left_right_index % 2 == 0:
                cochlea_1_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=left_ijk_apex,
                    cochlea_world_fiducials=cochlea_center_l_apex,
                    id_indextotal=self.id_indextotal_leftapex,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range,
                    beta_range=self.beta_range,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )

                self.id_indextotal_leftapex = self.id_indextotal_leftapex + 1

            else:
                cochlea_1_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=right_ijk_apex,
                    cochlea_world_fiducials=cochlea_center_r_apex,
                    id_indextotal=self.id_indextotal_rightapex,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range,
                    beta_range=self.beta_range,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )
                self.id_indextotal_rightapex = self.id_indextotal_rightapex + 1

            self.left_right_index = self.left_right_index + 1


        """##########class 2: cochlea basal  #####"""

        if class_id == 1:
            if self.left_right_index % 2 == 0:
                cochlea_2_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=left_ijk_origin,
                    cochlea_world_fiducials=cochlea_center_l_origin,
                    id_indextotal=self.id_indextotal_leftorigin,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range,
                    beta_range=self.beta_range,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )

                self.id_indextotal_leftorigin = self.id_indextotal_leftorigin + 1

            else:
                cochlea_2_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=right_ijk_origin,
                    cochlea_world_fiducials=cochlea_center_r_origin,
                    id_indextotal=self.id_indextotal_rightorigin,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range,
                    beta_range=self.beta_range,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )

                self.id_indextotal_rightorigin = self.id_indextotal_rightorigin + 1

            self.left_right_index = self.left_right_index + 1

            # """##########class 3: cochlea RW  #####"""

        if class_id == 2:
            if self.left_right_index % 2 == 0:
                cochlea_3_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=left_ijk_rw,
                    cochlea_world_fiducials=cochlea_center_l_rw,
                    id_indextotal=self.id_indextotal_leftrw,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range - 2,
                    beta_range=self.beta_range - 2,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )

                self.id_indextotal_leftrw = self.id_indextotal_leftrw + 1

            else:  # this 3
                cochlea_3_generator = cochleaclass_generator_withrotation_withoutaxial(
                    volume_name=volume_name,
                    landmark_ijk=right_ijk_rw,
                    cochlea_world_fiducials=cochlea_center_r_rw,
                    id_indextotal=self.id_indextotal_rightrw,
                    crop_manually_cache=self.crop_manually_cache,
                    crop_smallvolume_cache=self.crop_smallvolume_cache,
                    image_volume_rotator_distriutionajusted_volume_cache=self.image_volume_rotator_distriutionajusted_volume_cache,
                    single_classlen=single_classlen,
                    alpha_range=self.alpha_range - 2,
                    beta_range=self.beta_range - 2,
                    gamma_range=self.gamma_range,
                    steps1=self.steps1,
                    steps2=self.steps2,
                    steps3=self.steps3,
                    window_size=self.window_size,
                    paddinglist_clear=paddinglist_clear,
                    padding_crop_axial_list=self.padding_crop_axial_list,
                    padding_crop_coronal_list=self.padding_crop_coronal_list,
                    padding_crop_sagittal_list=self.padding_crop_sagittal_list,
                    gradual_rotated_padding_size=self.gradual_rotated_padding_size,
                    rotated_augmentation_size=self.rotated_augmentation_size,
                    indice=self.rotater_dicreset,
                    order=order,
                    justforiterationindex=justforiterationindex,
                )

                self.id_indextotal_rightrw = self.id_indextotal_rightrw + 1

            self.left_right_index = self.left_right_index + 1

            # """##########class 4: background #####"""

        if class_id == 3:
            background_generator = backgroundclass_generator_little(
                volume_name,
                left_ijk_origin,
                left_ijk_apex,
                left_ijk_rw,
                right_ijk_origin,
                right_ijk_apex,
                right_ijk_rw,
                self.selected_backgroundpoints,
                self.id_indextotal_background,
            )

            self.id_indextotal_background = self.id_indextotal_background + 1

        self.id_indextotal = self.id_indextotal + 1
        self.previous_klass = self.klass

        while True:
            try:
                print(f" start while true!!!! ")

                if class_id == 0:

                    print(
                        f"is class 1!and klass is {self.class_matrix[0]} class_id is {class_id} "
                    )

                    cochlea_1 = next(cochlea_1_generator)

                    assert (
                        cochlea_1.max() > 0.1
                    ), "start while true!!!!, outputing wrong cochlea!!"

                    cochlea_1 = tfimage(cochlea_1)

                    yield (cochlea_1, np.array(self.class_matrix[0]))
                if class_id == 1:

                    print(
                        f"is class 2!and klass is {self.class_matrix[1]} class_id is {class_id} "
                    )

                    cochlea_2 = next(cochlea_2_generator)

                    assert (
                        cochlea_2.max() > 0.1
                    ), "start while true!!!!, outputing wrong cochlea!!"

                    cochlea_2 = tfimage(cochlea_2)

                    yield (cochlea_2, np.array(self.class_matrix[1]))
                if class_id == 2:

                    print(
                        f"is class 3!and klass is {self.class_matrix[2]} and class_id is {class_id} "
                    )

                    cochlea_3 = next(cochlea_3_generator)

                    assert (
                        cochlea_3.max() > 0.1
                    ), "start while true!!!!, outputing wrong cochlea!!"

                    cochlea_3 = tfimage(cochlea_3)

                    yield (cochlea_3, np.array(self.class_matrix[2]))

                if class_id == 3:

                    print(
                        f"is class 4 background!and klass is {self.class_matrix[3]} class_id is {class_id}"
                    )

                    background_4 = next(background_generator)

                    background_4 = tfimage(background_4)

                    yield (background_4, np.array(self.class_matrix[3]))

            except StopIteration:
                print(f"StopIteration for class {self.klass}. Now going to next volume")
                break


