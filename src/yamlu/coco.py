import itertools
import json
import logging
import random
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
from joblib import delayed, Parallel
from joblib.externals import loky
from tqdm import tqdm

from yamlu.img import AnnotatedImage, Annotation
from yamlu.np_utils import to_python_type

_logger = logging.getLogger(__name__)

COCO_FIELD_KEYS = {"id", "category_id", "bbox", "image_id", "iscrowd", "keypoints"}

COCO_ANN_ID_FIELD = "coco_ann_id"


class Dataset(ABC):
    def __init__(
            self,
            dataset_path: Union[Path, str],
            split_n_imgs: Dict[str, int],
            coco_categories: List[Dict],
            keypoint_fields: List[str],
            relation_fields: List[str],
    ):
        """
        :param dataset_path: root of the dataset where images and coco json annotations are saved to
        :param split_n_imgs: number of imgs per split (e.g. {"train": 100, "valid": 5, "test": 5})
        :param coco_categories: coco categories with name, supercategory and id fields
        :param keypoint_fields: name of keypoint fields
        :param relation_fields: relation fields, i.e. name of Annotation fields that point to other Annotation objects
        """
        self.dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
        self.name = self.dataset_path.name

        self.split_n_imgs = split_n_imgs
        self.splits = list(self.split_n_imgs.keys())

        self.coco_categories = coco_categories
        self.cat_name_to_id = {c['name']: c['id'] for c in self.coco_categories}
        self.cat_id_to_name = {c['id']: c['name'] for c in self.coco_categories}

        self.keypoint_fields = keypoint_fields
        self.relation_fields = relation_fields

    def has_keypoints(self):
        return len(self.keypoint_fields) > 0

    @abstractmethod
    def get_split_ann_img(self, split: str, idx: int) -> AnnotatedImage:
        pass


class CocoDatasetExport:
    def __init__(
            self,
            ds: Dataset,
            write_img: bool = True,
            write_ann_img: bool = False,
            sample: Optional[int] = None,
            random_sample: bool = True,
            n_jobs: Optional[int] = None,
            ndigits: int = 3,
    ):
        """
        Args:
            write_ann_img: write image with annotated bounding boxes into an extra directory for debugging purposes
            random_sample: sample images randomly (True) or sample first k (False)
        """
        assert isinstance(write_img, bool)
        assert isinstance(write_ann_img, bool)

        self.write_img = write_img
        self.write_ann_img = write_ann_img

        self.ds = ds
        self.sample = sample
        self.random_sample = random_sample
        self.coco_json_exporter = CocoJsonExporter(self.ds, sample, ndigits)

        self.n_jobs = n_jobs
        if n_jobs is None:
            self.n_jobs = min([loky.backend.context.cpu_count() - 2, 16, max(ds.split_n_imgs.values())])
        _logger.info("Initialized %s with %d jobs", self.__class__.__name__, self.n_jobs)

    def dump_dataset(self) -> Dict[str, List[AnnotatedImage]]:
        return {split: self.dump_split(split) for split in self.ds.splits}

    def dump_split(self, split: str):
        _logger.info("%s: starting split=%s, write_img=%s, write_ann_img=%s, sample=%s, random_sample=%s", self.ds.name,
                     split, self.write_img, self.write_ann_img, self.sample, self.random_sample)
        assert split in self.ds.splits, f"{split} not in {self.ds.splits}"

        split_path = self.create_split_path_dir(split, remove_existing_images=self.write_img)

        ann_imgs_path = split_path.parent / f"{split}_annotated"
        if self.write_ann_img:
            ann_imgs_path.mkdir(exist_ok=True, parents=True)
            # remove existing files in directory
            for p in ann_imgs_path.iterdir():
                p.unlink()

        idxs = list(range(self.ds.split_n_imgs[split]))
        if self.sample is not None and len(idxs) > self.sample:
            if self.random_sample:
                random.seed(0)
                idxs = random.sample(idxs, self.sample)
            else:
                idxs = idxs[:self.sample]

        parallel = Parallel(n_jobs=self.n_jobs)
        ann_imgs = parallel(delayed(self.dump_image)(idx, split, split_path, ann_imgs_path) for idx in tqdm(idxs))

        self.coco_json_exporter.dump_split_coco_json(ann_imgs, split)

        return ann_imgs

    def dump_image(self, idx, split, split_path, ann_imgs_path):
        ann_img = self.ds.get_split_ann_img(split, idx)

        if self.write_img:
            img_path = split_path / ann_img.filename
            ann_img.img.save(img_path)

        if self.write_ann_img:
            ann_img.save_with_anns(ann_imgs_path)

        del ann_img.img

        return ann_img

    def create_split_path_dir(self, split, remove_existing_images: bool):
        ds_path = self.ds.dataset_path
        split_path = ds_path / split if self.sample is None else ds_path / f"sample_{self.sample}" / split

        split_path.mkdir(exist_ok=True, parents=True)

        if remove_existing_images:
            # remove existing files in directory
            for p in split_path.iterdir():
                p.unlink()

        return split_path


class CocoJsonExporter:
    def __init__(self, ds: Dataset, sample: Optional[int], ndigits: int):
        self.ds = ds
        self.sample = check_sample_param(sample)

        self._to_python_type = partial(to_python_type, ndigits=ndigits)

        self.excluded_fields = {COCO_ANN_ID_FIELD, *self.ds.keypoint_fields, *COCO_FIELD_KEYS, *self.ds.relation_fields}

    def dump_split_coco_json(self, ann_imgs: List[AnnotatedImage], split: str):
        indent = None if self.sample is None else 2

        coco_json_path = self._split_coco_json_path(split)
        _logger.info("Start dumping coco %d annotations to %s", len(ann_imgs), coco_json_path)
        coco = self.create_coco_dict(ann_imgs)
        _logger.info("Finished creating coco annotations")

        coco_json_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            with coco_json_path.open("w") as f:
                json.dump(coco, f, indent=indent)
        except TypeError as e:
            _logger.warning("Coco object not json serializable:\n%s", coco)
            raise e

    def create_coco_dict(self, ann_imgs: List[AnnotatedImage]) -> Dict:
        coco = {
            "images": _create_images_dict(ann_imgs),
            "annotations": self._create_annotations_dict(ann_imgs),
            "categories": self.ds.coco_categories
        }
        return coco

    def _split_coco_json_path(self, split: str) -> Path:
        if self.sample is None:
            return self.ds.dataset_path / f"{split}.json"
        else:
            return self.ds.dataset_path / f"sample_{self.sample}" / f"{split}.json"

    def _create_img_anns(self, t: Tuple[int, AnnotatedImage]) -> List[Dict]:
        img_id, ann_img = t

        # create artifical unique annotation id
        # assuming that an image has less than 1000 annotations
        assert len(ann_img.annotations) < 1000

        # set id field first so that they can be used for relation mappings
        for i, ann in enumerate(ann_img.annotations):
            ann.set(COCO_ANN_ID_FIELD, int(img_id * 1000 + i))

        coco_anns = [self._create_coco_ann(ann, img_id) for ann in ann_img.annotations]
        return coco_anns

    def _create_coco_ann(self, ann: Annotation, img_id: int) -> Dict:

        coco_ann = {
            "id": ann.get(COCO_ANN_ID_FIELD),
            "image_id": int(img_id),
            "category": ann.category,  # this isn't strictly required but makes debugging easier
            "category_id": self.ds.cat_name_to_id[ann.category],
            "area": self._to_python_type(ann.bb.area),  # needed by coco eval
            "bbox": list(map(self._to_python_type, ann.bb.bb_coco)),
            "iscrowd": 0,
        }

        if self.ds.has_keypoints():
            # in coco format convention keypoints have an additional visibility flag
            # - v=0: not labeled (in which case x=y=0)
            # - v=1: labeled but not visible
            # - v=2: labeled and visible
            # see: http://cocodataset.org/#format-data
            # this means each keypoint is represented as [x,y,v]

            # frameworks like detectron2 assume that if the dataset has keypoints, this applies to all categories
            # as a workaround we create v=0 keypoints for categories without keypoints
            kps = np.zeros(3 * len(self.ds.keypoint_fields))

            for i, k in enumerate(self.ds.keypoint_fields):
                if k in ann:
                    x, y = getattr(ann, k)
                    kps[i * 3: i * 3 + 3] = [x, y, 2]

            coco_ann["keypoints"] = self._to_python_type(kps)

        for rel_name in self.ds.relation_fields:
            if rel_name in ann:
                # relationship fields point to other Annotation objects
                rel_ann = ann.get(rel_name)
                coco_ann[rel_name] = rel_ann.get(COCO_ANN_ID_FIELD)

        # make sure we don't overwrite reserved fields
        coco_ann.update({
            k: self._to_python_type(v) for k, v in ann.extra_fields.items() if
            k not in self.excluded_fields and not isinstance(v, Annotation)
        })
        return coco_ann

    def _create_annotations_dict(self, ann_imgs: List[AnnotatedImage]):
        imgs_anns = [self._create_img_anns((i, ai)) for i, ai in enumerate(tqdm(ann_imgs))]
        flat_anns = list(itertools.chain(*imgs_anns))
        return flat_anns


def check_sample_param(sample):
    assert sample is None or not isinstance(sample, bool), f"sample is not None or an int: {sample} ({type(sample)})"
    return sample


def _create_images_dict(ann_imgs: List[AnnotatedImage]):
    return [
        {
            "file_name": ann_img.filename,
            "height": int(ann_img.height),
            "width": int(ann_img.width),
            "id": idx
        } for idx, ann_img in enumerate(ann_imgs)
    ]
