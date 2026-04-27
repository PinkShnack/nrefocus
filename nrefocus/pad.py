"""Convenience functions for padding

.. versionadded:: 0.1.4
"""
from __future__ import division, print_function

from ._ndarray_backend import xp


def _get_pad_left_right(small, large):
    """Compute left and right padding values.

    Here we use the convention that if the padding
    size is odd, we pad the odd part to the right
    and the even part to the left.

    Parameters
    ----------
    small: int
        Old size of original 1D array
    large: int
        New size off padded 1D array

    Returns
    -------
    (padleft, padright) : tuple
        The proposed padding sizes.
    """
    assert small < large, "Can only pad when new size larger than old size"

    padsize = large - small
    if padsize % 2 != 0:
        leftpad = (padsize - 1)/2
    else:
        leftpad = padsize/2
    rightpad = padsize-leftpad

    return int(leftpad), int(rightpad)


def pad_add(av, size=None, stlen=10):
    """ Perform linear padding for complex array

    The input array `av` is padded with a linear ramp starting at the
    edges and going outwards to an average value computed from a band
    of thickness `stlen` at the outer boundary of the array.

    Pads will only be appended, not prepended to the array.

    If the input array is complex, pads will be complex numbers
    The average is computed for phase and amplitude separately.

    Parameters
    ----------
    av: complex 1D or 2D ndarray
        The array that will be padded.
    size: int or tuple of length 1 (1D) or tuple of length 2 (2D), optional
        The final size of the padded array. Defaults to double the size
        of the input array.
    stlen: int, optional
        The thickness of the frame within `av` that will be used to
        compute an average value for padding.


    Returns
    -------
    pv: complex 1D or 2D ndarray
        Padded array `av` with pads appended to right and bottom.
    """
    if size is None:
        if len(av.shape) == 1:
            size = [int(2 * av.shape[0])]
        else:
            # For stacks (..., y, x), pad only spatial axes.
            size = [int(2 * av.shape[-2]), int(2 * av.shape[-1])]
    elif not hasattr(size, "__len__"):
        if len(av.shape) == 1:
            size = [size]
        else:
            size = [size, size]

    if len(av.shape) == 1:
        assert len(size) == 1
        return _pad_add_1d(av, size, stlen)

    # 2D or stack of 2D: (..., y, x)
    if len(size) == len(av.shape):
        # Backwards-compatible input: ignore leading dims but require
        # they are not being "padded" to a different size.
        assert list(size[:-2]) == list(av.shape[:-2]), (
            "For stacks, only the last two axes can be padded; leading "
            "sizes must match `av.shape`."
        )
        size = list(size[-2:])

    assert len(size) == 2, (
        "`size` must be a scalar or a tuple/list of length 2 for 2D inputs "
        "or stacks (..., y, x)."
    )

    if len(av.shape) == 2:
        return _pad_add_2d(av, size, stlen)

    # Stack: pad each 2D slice independently to preserve legacy behavior
    # (i.e. same results as calling pad_add on each slice).
    lead_shape = av.shape[:-2]
    ny, nx = av.shape[-2], av.shape[-1]
    out_y, out_x = int(size[0]), int(size[1])
    av2 = av.reshape((-1, ny, nx))
    out2 = xp.empty((av2.shape[0], out_y, out_x), dtype=av.dtype)
    for i in range(av2.shape[0]):
        out2[i] = _pad_add_2d(av2[i], [out_y, out_x], stlen)
    return out2.reshape(lead_shape + (out_y, out_x))


def _pad_add_1d(av, size, stlen):
    """1D component of `pad_add`"""
    assert len(size) == 1

    padx = _get_pad_left_right(av.shape[0], size[0])

    mask = xp.zeros(av.shape, dtype=bool)
    mask[stlen:-stlen] = True
    border = av[~mask]
    if av.dtype.name.count("complex"):
        padval = xp.average(xp.abs(border)) * \
            xp.exp(1j*xp.average(xp.angle(border)))
    else:
        padval = xp.average(border)
    if xp.__version__[:3] in ["1.7", "1.8", "1.9"]:
        end_values = ((padval, padval),)
    else:
        end_values = (padval,)
    bv = xp.pad(av,
                padx,
                mode="linear_ramp",
                end_values=end_values)
    # roll the array so that the padding values are on the right
    bv = xp.roll(bv, -padx[0], 0)
    return bv


def _pad_add_2d(av, size, stlen):
    """2D component of `pad_add`"""
    assert len(size) == 2

    padx = _get_pad_left_right(av.shape[0], size[0])
    pady = _get_pad_left_right(av.shape[1], size[1])

    mask = xp.zeros(av.shape, dtype=bool)
    mask[stlen:-stlen, stlen:-stlen] = True
    border = av[~mask]
    if av.dtype.name.count("complex"):
        padval = xp.average(xp.abs(border)) * \
            xp.exp(1j*xp.average(xp.angle(border)))
    else:
        padval = xp.average(border)
    padval = padval.item()  # with cupy 0d array we have to get the value
    if xp.__version__[:3] in ["1.7", "1.8", "1.9"]:
        end_values = ((padval, padval), (padval, padval))
    else:
        end_values = (padval,)
    bv = xp.pad(av,
                (padx, pady),
                mode="linear_ramp",
                end_values=end_values)
    # roll the array so that the padding values are on the right
    bv = xp.roll(bv, -padx[0], 0)
    bv = xp.roll(bv, -pady[0], 1)
    return bv


def pad_rem(pv, size=None):
    """Removes linear padding from array

    This is a convenience function that does the opposite
    of `pad_add`.

    Parameters
    ----------
    pv: 1D or 2D ndarray
        The array from which the padding will be removed.
    size: tuple of length 1 (1D) or 2 (2D), optional
        The final size of the un-padded array. Defaults to half the size
        of the input array.


    Returns
    -------
    pv: 1D or 2D ndarray
        Padded array `av` with pads appended to right and bottom.
    """
    if size is None:
        if len(pv.shape) == 1:
            assert pv.shape[0] % 2 == 0, (
                "Uneven size; specify correct size of output!"
            )
            size = [int(pv.shape[0] / 2)]
        else:
            # For stacks (..., y, x), remove padding only on spatial axes.
            assert pv.shape[-2] % 2 == 0 and pv.shape[-1] % 2 == 0, (
                "Uneven size; specify correct size of output!"
            )
            size = [int(pv.shape[-2] / 2), int(pv.shape[-1] / 2)]
    elif not hasattr(size, "__len__"):
        if len(pv.shape) == 1:
            size = [size]
        else:
            size = [size, size]

    if len(pv.shape) == 1:
        assert len(size) == 1
        return pv[:size[0]]

    if len(size) == len(pv.shape):
        assert list(size[:-2]) == list(pv.shape[:-2]), (
            "For stacks, only the last two axes can be unpadded; leading "
            "sizes must match `pv.shape`."
        )
        size = list(size[-2:])

    assert len(size) == 2, (
        "`size` must be a scalar or a tuple/list of length 2 for 2D inputs "
        "or stacks (..., y, x)."
    )

    # Works for 2D and stacks: keep leading axes, crop last two.
    return pv[..., :size[0], :size[1]]
