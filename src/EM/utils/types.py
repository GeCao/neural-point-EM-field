from enum import Enum


class MessageAttribute(Enum):
    EInfo = 0
    EWarn = 1
    EError = 2

    def __int__(self):
        return self.value


class ModuleType(Enum):
    LIGHTFIELD = 0

    def __int__(self):
        return self.value


class NodeType(Enum):
    BACKGROUND = 0

    def __int__(self):
        return self.value


class TrainType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2

    def __int__(self):
        return self.value


class FeatureWeighting(Enum):
    ATTENTION = 0
    LINEAR = 1
    MAXPOOL = 2
    SUM = 3

    def __int__(self):
        return self.value


class LearnTarget(Enum):
    RECEIVER_GAIN = 0
    COVERAGE_MAP = 1

    def __int__(self):
        return self.value
