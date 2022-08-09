import pytest

from pybasic.utils import _validate_iter_axes


class TestValidateIterDims2D:
    def test_valid(self):
        iter_axes = _validate_iter_axes((5, 5), rgb=False, iter_axes=set())
        assert iter_axes == set()

    def test_rgb(self):
        with pytest.warns(UserWarning):
            iter_axes = _validate_iter_axes((5, 5), rgb=True, iter_axes=set())
        assert iter_axes == set()

    def test_iter_axes(self):
        with pytest.warns(UserWarning):
            iter_axes = _validate_iter_axes((5, 5), rgb=False, iter_axes={0})
        assert iter_axes == set()


class TestValidateIterDims3D:
    def test_no_axes(self):
        with pytest.raises(ValueError):
            _validate_iter_axes((5, 5, 5), rgb=False, iter_axes=set())

    def test_with_axes(self):
        iter_axes = _validate_iter_axes((5, 5, 5), rgb=False, iter_axes={0})
        assert iter_axes == {0}

    def test_rgb(self):
        iter_axes = _validate_iter_axes((3, 5, 5), rgb=True, iter_axes=set())
        assert iter_axes == {0}

    def test_invalid_axes(self):
        with pytest.raises(ValueError):
            _validate_iter_axes((5, 5, 5), rgb=False, iter_axes={3})
