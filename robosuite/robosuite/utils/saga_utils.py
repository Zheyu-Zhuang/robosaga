import os
import shutil
import xml.etree.ElementTree as ET

from robosuite.models.objects import (
    BottleObject,
    BreadObject,
    CanObject,
    CerealObject,
    LemonObject,
    MilkObject,
)


def distractors_to_model(distractors):
    if distractors is None:
        return []
    supported_distractors = {
        "bottle": BottleObject,
        "lemon": LemonObject,
        "milk": MilkObject,
        "bread": BreadObject,
        "can": CanObject,
        "cereal": CerealObject,
    }
    idx = 0
    models = []
    for distractor_ in distractors:
        if distractor_ not in supported_distractors.keys():
            raise ValueError(
                "Distractor {} not supported. Supported distractors are {}".format(
                    distractor_, supported_distractors.keys()
                )
            )
        else:
            name = "distractor_{}".format(idx)
            models.append(supported_distractors[distractor_](name=name))
            idx += 1
    return models


def replace_texture(xml_file, new_texture_path, texture_name="table_ceramic"):
    # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define the new texture element
    new_texture = ET.Element(
        "texture", attrib={"file": new_texture_path, "type": "2d", "name": "new_texture"}
    )

    # Insert the new texture element into the <asset> section
    asset = root.find("asset")
    asset.append(new_texture)

    # Find the material for the table and update the texture reference
    for material in asset.findall("material"):
        if material.get("name") == texture_name:
            material.set("texture", "new_texture")
    texture_file_name = get_texture_name(new_texture_path)

    xml_file = xml_file.replace(".xml", f"_{texture_file_name}_temp.xml")
    # Save the modified XML to a new file or overwrite the existing one
    tree.write(xml_file)
    return xml_file


def get_texture_name(texture_path):
    texture_file_name = os.path.basename(texture_path)
    return texture_file_name.split(".")[0]
