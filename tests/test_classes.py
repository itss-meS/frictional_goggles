from pathlib import Path

import yaml

from src.classes import CAMO_CLASSES, CLASS_NAMES, group_of, is_camo


def test_data_yaml_matches_class_names():
    data = yaml.safe_load((Path(__file__).parent.parent / "configs" / "data.yaml").read_text())
    names = [data["names"][i] for i in range(len(data["names"]))]
    assert names == CLASS_NAMES


def test_groups():
    assert group_of("soldier_desert") == "person"
    assert group_of("person") == "person"
    assert group_of("vehicle_camo_net") == "vehicle"
    assert group_of("truck") == "vehicle"
    assert group_of("drone_camo") == "drone"
    assert group_of("dog") == "dog"


def test_camo_flags():
    assert CAMO_CLASSES <= set(CLASS_NAMES)
    assert is_camo("soldier_snow")
    assert not is_camo("person")
