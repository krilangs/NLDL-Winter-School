"""
    Generic data loading routines for the SEN12MS dataset of corresponding Sentinel 1,
    Sentinel 2 and Modis LandCover data.

    The SEN12MS class is meant to provide a set of helper routines for loading individual
    image patches as well as triplets of patches from the dataset. These routines can easily
    be wrapped or extended for use with many deep learning frameworks or as standalone helper
    methods. For an example use case please see the "main" routine at the end of this file.

    NOTE: Some folder/file existence and validity checks are implemented but it is
          by no means complete.

    Author: Lloyd Hughes (lloyd.hughes@tum.de)
"""

import os
import rasterio

import numpy as np

from enum import Enum
from glob import glob
import torch
from einops import rearrange
from itertools import product
import PIL

class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []


class LCBands(Enum):
    IGBP = igbp = 1
    LCCS1 = landcover = 2
    LCCS2 = landuse = 3
    LCCS3 = hydrology = 4
    ALL = [IGBP, LCCS1, LCCS2, LCCS3]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    lc = "lc"

# Note: The order in which you request the bands is the same order they will be returned in.


class SEN12MSDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            raise Exception(
                "The specified base_dir for SEN12MS dataset does not exist")

    """
        Returns a list of scene ids for a specific season.
    """

    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season)

        if not os.path.exists(path):
            raise NameError("Could not find season {} in base directory {}".format(
                season, self.base_dir))

        scene_list = [os.path.basename(s)
                      for s in glob(os.path.join(path, "*"))]
        scene_list = [int(s.split('_')[1]) for s in scene_list]
        return set(scene_list)

    """
        Returns a list of patch ids for a specific scene within a specific season
    """

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season, f"s2_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids

    """
        Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
    """

    def get_season_ids(self, season):
        season = Seasons(season).value
        ids = {}
        scene_ids = self.get_scene_ids(season)
        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    """
        Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
    """

    def get_patch(self, season, scene_id, patch_id, bands):
        season = Seasons(season).value
        sensor = None

        if isinstance(bands, (list, tuple)):
            b = bands[0]
        else:
            b = bands

        if isinstance(b, S1Bands):
            sensor = Sensor.s1.value
            bandEnum = S1Bands
        elif isinstance(b, S2Bands):
            sensor = Sensor.s2.value
            bandEnum = S2Bands
        elif isinstance(b, LCBands):
            sensor = Sensor.lc.value
            bandEnum = LCBands
        else:
            raise Exception("Invalid bands specified")

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value

        scene = "{}_{}".format(sensor, scene_id)
        filename = "{}_{}_p{}.tif".format(season, scene, patch_id)
        patch_path = os.path.join(self.base_dir, season, scene, filename)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        return data, bounds

    """
        Returns a triplet of patches. S1, S2 and LC as well as the geo-bounds of the patch
    """

    def get_s1s2lc_triplet(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL):
        s1, bounds = self.get_patch(season, scene_id, patch_id, s1_bands)
        s2, _ = self.get_patch(season, scene_id, patch_id, s2_bands)
        lc, _ = self.get_patch(season, scene_id, patch_id, lc_bands)

        return s1, s2, lc, bounds

    """
        Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or LC
    """

    def get_triplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, lc_bands=LCBands.ALL):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        lc_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, lc, bound = self.get_s1s2lc_triplet(
                    season, sid, pid, s1_bands, s2_bands, lc_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                lc_data.append(lc)
                bounds.append(bound)

        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(lc_data, axis=0), bounds


class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self, directory):
        self.sen12ms = SEN12MSDataset(directory)
        self.summer_ids = self.sen12ms.get_season_ids(Seasons.SUMMER)

        self.scene_patch_list = []
        for x in self.summer_ids.items():
            self.scene_patch_list.extend(list(product([x[0]],x[1])))

        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.scene_patch_list)

    def __getitem__(self, idx):

        SCENE_ID, PATCH_ID = self.scene_patch_list[idx]

        s2_rgb_patch, bounds = self.sen12ms.get_patch(Seasons.SUMMER, SCENE_ID,
                                                PATCH_ID, bands=S2Bands.RGB)

        data = np.clip(np.power(s2_rgb_patch/5000, 1/2.2)*255, 0, 255).astype(np.uint8)
        data = PIL.Image.fromarray(rearrange(data, 'C X Y -> X Y C'))

        if self.transform:
            data = self.transform(data)

        return {'image': data}
