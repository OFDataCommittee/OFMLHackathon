
from pytest import raises
from ..environment import Environment


def test_environment():
    with raises(TypeError):
        env = Environment("", "", "", "", "", 1, 1, 1)

