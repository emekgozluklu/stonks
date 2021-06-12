import os


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
