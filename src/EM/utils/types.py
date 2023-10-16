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
    Train = 0
    TEST = 1
    VALIDATION = 2

    def __int__(self):
        return self.value
