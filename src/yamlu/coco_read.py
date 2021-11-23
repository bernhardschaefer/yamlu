import json
from collections import defaultdict
from pathlib import Path
from typing import List, Set

import numpy as np
from tqdm import tqdm

from yamlu.img import AnnotatedImage, Annotation, BoundingBox

ARROW_RELATIONS = ["arrow_prev", "arrow_next"]


class CocoReader:
    """
    Parses diagram datasets stored in COCO format (https://cocodataset.org/#format-data).
    The COCO format is extended with some special relation fields for arrows and text.
    """

    def __init__(self, dataset_root: Path, arrow_categories: Set[str] = ("arrow",)):
        self.dataset_root = dataset_root
        self.arrow_categories = set(arrow_categories)
        self.splits = [p.name for p in self.dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    def parse_split(self, split: str) -> List[AnnotatedImage]:
        json_path = self.dataset_root / f"{split}.json"
        imgs_path = self.dataset_root / split

        coco = json.loads(json_path.read_text())
        image_id_to_anns = defaultdict(list)
        for d in coco["annotations"]:
            image_id_to_anns[d["image_id"]].append(d)

        ais = []
        for coco_image in tqdm(coco["images"]):
            coco_anns = image_id_to_anns[coco_image["id"]]
            anns = self._parse_coco_anns(coco_anns)
            img_path = imgs_path / coco_image["file_name"]
            ai = AnnotatedImage.from_img_path(img_path, anns)
            ais.append(ai)

        return ais

    def _parse_coco_anns(self, coco_anns) -> List[Annotation]:
        anns = []
        for coco_ann in coco_anns:
            a = Annotation(
                category=coco_ann["category"],
                bb=BoundingBox.from_xywh(*coco_ann["bbox"]),
                id=coco_ann["id"],
            )
            # parse keypoints for arrows
            # In COCO format, "keypoints" is a length 3k array
            # Each keypoint has a location x,y and a visibility flag v defined as:
            # - v=0: not labeled (in which case x=y=0)
            # - v=1: labeled but not visible
            # - v=2: labeled and visible.
            # In the diagram datasets all arrow keypoints are v=2 "labeled and visible", so we can ignore this flag
            if a.category in self.arrow_categories:
                a.keypoints = np.array(coco_ann["keypoints"]).reshape(-1, 3)
            anns.append(a)

        id_to_ann = {a.id: a for a in anns}

        # parse arrow source and target relations
        coco_arrows = [a for a in coco_anns if a["category"] in self.arrow_categories]
        for coco_arw in coco_arrows:
            a = id_to_ann[coco_arw["id"]]
            for rel in ARROW_RELATIONS:
                if rel in coco_arw.keys():
                    rel_shape = id_to_ann[coco_arw[rel]]
                    a.set(rel, rel_shape)

        return anns
