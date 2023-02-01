
from typing import Any
from os.path import isdir, isfile, basename, join
from glob import glob
import fileinput
import sys


def get_time_folders(path: str):
    def is_float(element: Any) -> bool:
        # taken from: https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
        try:
            float(element)
            return True
        except ValueError:
            return False
    folders = [basename(x) for x in glob(join(path, "[0-9]*"))
               if isdir(x) and is_float(basename(x))]
    return folders


def get_latest_time(path: str) -> str:
    folders = get_time_folders(path)
    if not folders:
        if isdir(join(path, "0.org")):
            return "0.org"
        else:
            raise ValueError(f"Could not find time folder in {path}")
    return sorted(folders, key=float)[-1]


def fetch_line_from_file(path: str, keyword: str) -> str and list:
    with open(path) as f:
        lines, idx = [], []
        for i, line in enumerate(f.readlines()):
            if keyword in line:
                lines.append(line)
                idx.append(i)
        return lines if len(lines) > 1 else lines[0], idx


def replace_line_in_file(path: str, keyword: str, new: str):
    """Keyword-based replacement of one or more lines in a file.

    :param path: file location
    :type path: str
    :param keyword: keyword based on which lines are selected
    :type keyword: str
    :param new: the new line replacing the old one
    :type new: str
    """
    new = new + "\n" if not new.endswith("\n") else new
    fin = fileinput.input(path, inplace=True)
    for line in fin:
        if keyword in line:
            line = new
        sys.stdout.write(line)
    fin.close()


def replace_line_latest(path: str, filename: str, keyword: str, new: str,
                        processor: bool = True):
    search_path = join(path, "processor0") if processor else path
    latest_time = get_latest_time(search_path)
    if processor:
        for p in glob(join(path, "processor*")):
            replace_line_in_file(
                join(p, latest_time, filename), keyword, new
            )
    else:
        replace_line_in_file(
            join(path, latest_time, filename), keyword, new
        )


def check_path(path: str):
    if not isdir(path):
        raise ValueError(f"Could not find path {path}")


def check_file(file_path: str):
    if not isfile(file_path):
        raise ValueError(f"Could not find file {file_path}")


def check_pos_int(value: int, name: str, with_zero=False):
    message = f"Argument {name} must be a positive integer; got {value}"
    if not isinstance(value, int):
        raise ValueError(message)
    lb = 0 if with_zero else 1
    if value < lb:
        raise ValueError(message)


def check_pos_float(value: float, name: str, with_zero=False):
    message = f"Argument {name} must be a positive float; got {value}"
    if not isinstance(value, (float, int)):
        raise ValueError(message)
    if with_zero and value < 0.0:
        raise ValueError(message)
    if not with_zero and value <= 0.0:
        raise ValueError(message)
