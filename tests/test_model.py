import pytest

from dueling_arch.models.model import Network


def test_model():
    assert Network()(2) == 4
