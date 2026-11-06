CLASS_NAMES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "boat",
    "drone",
    "soldier_woodland",
    "soldier_desert",
    "soldier_urban",
    "soldier_snow",
    "vehicle_camo_net",
    "static_camo_net",
    "drone_camo",
]

CAMO_CLASSES = frozenset(
    {
        "soldier_woodland",
        "soldier_desert",
        "soldier_urban",
        "soldier_snow",
        "vehicle_camo_net",
        "static_camo_net",
        "drone_camo",
    }
)

GROUPS = {
    "person": "person",
    "soldier_woodland": "person",
    "soldier_desert": "person",
    "soldier_urban": "person",
    "soldier_snow": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "vehicle_camo_net": "vehicle",
    "drone": "drone",
    "drone_camo": "drone",
}


def group_of(name: str) -> str:
    return GROUPS.get(name, name)


def is_camo(name: str) -> bool:
    return name in CAMO_CLASSES
