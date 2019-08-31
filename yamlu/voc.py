import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from yamlu.img import AnnotatedImage, BoundingBox, Annotation


def parse_voc_xml_img(img_path: Path, voc_xml: Union[str, Path]) -> AnnotatedImage:
    tree = ET.parse(voc_xml)
    root = tree.getroot()

    filename = root.find('filename').text

    anns = []
    for obj in root.iter('object'):
        category = obj.find("name").text

        bbox = obj.find("bndbox")
        bb_dict = dict((e.tag, int(e.text)) for e in bbox.getchildren())
        # we leverage the fact that voc uses xmin, ymin, xmax, ymax field names
        bb = BoundingBox.from_xyxy(**bb_dict)

        anns.append(Annotation(category=category, bb=bb))

    img = Image.open(img_path)
    img_np = np.array(img)
    return AnnotatedImage(filename, width=img.width, height=img.height, annotations=anns, img=img_np)
