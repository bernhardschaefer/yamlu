import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

from yamlu.img import AnnotatedImage, BoundingBox, Annotation


def parse_voc_annotations(voc_xml_path: Union[str, Path]):
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()

    # filename = root.find('filename').text

    anns = []
    for obj in root.iter('object'):
        category = obj.find("name").text

        bbox = obj.find("bndbox")
        bb_dict = {e.tag: int(e.text) for e in bbox}
        # we leverage the fact that voc uses xmin, ymin, xmax, ymax field names
        bb = BoundingBox.from_pascal_voc(*bb_dict.values())

        anns.append(Annotation(category=category, bb=bb))

    return anns


def parse_voc_xml_img(img_path: Path, voc_xml_path: Union[str, Path]) -> AnnotatedImage:
    annotations = parse_voc_annotations(voc_xml_path)
    return AnnotatedImage.from_img_path(img_path, annotations)
