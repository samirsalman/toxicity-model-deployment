from enum import Enum


class ModelStatus(Enum):
    STOPPED="stopped"
    STARTING="starting"
    RUNNING="running"
    ERROR="error"