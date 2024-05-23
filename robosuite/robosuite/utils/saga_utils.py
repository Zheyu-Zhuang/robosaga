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
