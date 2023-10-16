import os


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
