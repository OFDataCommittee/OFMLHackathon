
from os import remove, makedirs
from os.path import join
from shutil import rmtree
from ..utils import (get_latest_time, replace_line_in_file,
                     replace_line_latest, fetch_line_from_file)


def test_get_latest_time():
    test_folder = join("/tmp", "test_get_latest_time")
    folders = ["constant", "system", "0.org", "1e-6", "0", "1", "1.01"]
    for f in folders:
        makedirs(join(test_folder, f), exist_ok=True)
    latest = get_latest_time(test_folder)
    rmtree(test_folder)
    assert latest == "1.01"


def test_fetch_line_from_file():
    testfile = "/tmp/test_replace_line_in_file.txt"
    with open(testfile, "w+") as tf:
        tf.write(
            """
policy model;
seed 1;
train true;
            """
        )
    line = fetch_line_from_file(testfile, "train")
    remove(testfile)
    assert "true" in line


def test_replace_line_in_file():
    testfile = "/tmp/test_replace_line_in_file.txt"
    with open(testfile, "w+") as tf:
        tf.write(
            """
policy model;
seed 1;
train true;
            """
        )
    keyword, new = "seed", "seed 0;"
    replace_line_in_file(testfile, keyword, new)
    with open(testfile, "r") as tf:
        found_new_line = False
        for line in tf.readlines():
            if new in line:
                found_new_line = True
    remove(testfile)
    assert found_new_line


def test_replace_line_latest():
    test_folder = join("/tmp", "test_replace_latest")
    for p in ["processor0", "processor1"]:
        makedirs(join(test_folder, p, "2"), exist_ok=True)
        with open(join(test_folder, p, "2", "U"), "w+") as tf:
            tf.write("seed 0;")
    replace_line_latest(test_folder, "U", "seed", "seed 1;")
    found = 0
    for p in ["processor0", "processor1"]:
        with open(join(test_folder, p, "2", "U"), "r") as tf:
            for line in tf.readlines():
                if "seed 1;" in line:
                    found += 1
    rmtree(test_folder)
    assert found == 2
