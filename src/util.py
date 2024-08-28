import traceback
import os
import shutil
import math


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def handle_exception(e):
    print(e)
    print(traceback.format_exc())


def copy_files(files, source_dir, destination_dir):
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
        else:
            print(f"warning {source_file} not exist")


def normal_val(x, level=10):
    return math.tanh(-10 * x) + 1
