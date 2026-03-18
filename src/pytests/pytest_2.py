import pytest
import subprocess


@pytest.fixture
def test_data():
    return [5, 8, 2, 9, 6, 3]


def argmin(liste):
    if len(liste) == 0:
        return None

    idx_min = 0
    value_min = liste[0]
    for i, x in enumerate(liste):
        if x < value_min:
            value_min = x
            idx_min = i
    return idx_min


def argmax(liste):
    if len(liste) == 0:
        return None

    idx_max = 0
    value_max = liste[0]
    for i, x in enumerate(liste):
        if x > value_max:
            value_max = x
            idx_max = i
    return idx_max


def test_argmin(test_data):
    assert argmin(test_data) == 2


def test_argmax(test_data):
    assert argmax(test_data) == 3


#! pytest /tmp/pytest_2.py -v
subprocess.call(["pytest", "/tmp/pytest_2.py", "-v"])
