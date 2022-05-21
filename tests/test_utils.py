import pytest

from pybasic.utils import _validate_iter_dims


class TestValidateIterDims2D:
    def test_valid(self):
        iter_axes = _validate_iter_dims((5, 5), rgb=False, iter_dims=set())
        assert iter_axes == set()

    def test_rgb(self):
        with pytest.warns(UserWarning):
            iter_axes = _validate_iter_dims((5, 5), rgb=True, iter_dims=set())
        assert iter_axes == set()

    def test_iter_axes(self):
        with pytest.warns(UserWarning):
            iter_axes = _validate_iter_dims((5, 5), rgb=False, iter_dims={0})
        assert iter_axes == set()


class TestValidateIterDims3D:
    def test_no_axes(self):
        with pytest.raises(ValueError):
            _validate_iter_dims((5, 5, 5), rgb=False, iter_dims=set())

    def test_with_axes(self):
        iter_axes = _validate_iter_dims((5, 5, 5), rgb=False, iter_dims={0})
        assert iter_axes == {0}

    def test_rgb(self):
        iter_axes = _validate_iter_dims((3, 5, 5), rgb=True, iter_dims=set())
        assert iter_axes == {0}

    def test_invalid_axes(self):
        with pytest.raises(ValueError):
            _validate_iter_dims((5, 5, 5), rgb=False, iter_dims={3})
