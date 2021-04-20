import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Set
from xml.dom import minidom

from yamlu.img import AnnotatedImage, BoundingBox, Annotation
from yamlu.np_utils import to_python_type


def dump_ai_voc(ai: AnnotatedImage, folder: Path, additional_fields: Set = None) -> Path:
    additional_fields = {} if additional_fields is None else additional_fields

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder.name
    ET.SubElement(root, "filename").text = ai.filename
    ET.SubElement(root, "path").text = str(folder / ai.filename)

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(ai.width)
    ET.SubElement(size, "height").text = str(ai.height)
    # ET.SubElement(size, "depth").text = ai.height

    for a in ai.annotations:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = a.category
        ET.SubElement(obj, "difficult").text = str(0)
        ET.SubElement(obj, "occluded").text = str(0)

        for k in additional_fields:
            if k in a:
                ET.SubElement(obj, k).text = str(getattr(a, k))

        bndbox = ET.SubElement(obj, "bndbox")
        bb = a.bb
        # Pascal annotations pixel-based integers in the range [1, W or H],
        # where a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        for k, v in zip(["xmin", "ymin", "xmax", "ymax"], [bb.l + 1, bb.t + 1, bb.r, bb.b]):
            ET.SubElement(bndbox, k).text = str(to_python_type(v, 0))

    folder.mkdir(parents=True, exist_ok=True)
    xml_path = folder / f"{ai.fname_without_suffix}.xml"
    # tree = ET.ElementTree(root)
    # tree.write(str(xml_path), encoding="utf-8")

    # without dependencies like lxml I have to reparse wiht minidom to get pretty xml :-/
    # https://stackoverflow.com/questions/28813876/how-do-i-get-pythons-elementtree-to-pretty-print-to-an-xml-file
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    xml_path.write_text(xmlstr)
    return xml_path


def parse_voc_annotations(voc_xml_path: Union[str, Path]):
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()
    # filename = root.find('filename').text

    anns = []
    for obj in root.iter('object'):
        category = obj.find("name").text

        bbox = obj.find("bndbox")
        bb_dict = {e.tag: float(e.text) for e in bbox}
        # we leverage the fact that voc uses xmin, ymin, xmax, ymax field names
        bb = BoundingBox.from_pascal_voc(**bb_dict)

        # Note: difficult and occluded is not parsed since I don't use them at the moment
        add_elements = [e for e in obj if e.tag not in ["name", "bndbox", "difficult", "occluded"]]
        add_fields = {e.tag: e.text for e in add_elements}

        anns.append(Annotation(category=category, bb=bb, **add_fields))

    return anns


def parse_voc_xml_img(img_path: Path, voc_xml_path: Union[str, Path]) -> AnnotatedImage:
    annotations = parse_voc_annotations(voc_xml_path)
    return AnnotatedImage.from_img_path(img_path, annotations)
