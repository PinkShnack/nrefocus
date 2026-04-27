import pathlib

import numpy as np

import nrefocus

from ..helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent.parent / "data"


@skip_if_missing("cupy")
def test_2d_refocus1_cupy(set_ndarray_backend_to_cupy):
    pixel_size = 1e-6
    rf = nrefocus.RefocusCupy(field=np.arange(256).reshape(16, 16),
                              wavelength=8.25 * pixel_size,
                              pixel_size=pixel_size,
                              medium_index=1.533,
                              distance=0,
                              kernel="helmholtz",
                              padding=False)

    refocused = rf.propagate(distance=2.13 * pixel_size)
    refocused_cpu = refocused.get()
    reference = np.loadtxt(data_path / "test_2d_refocus1.txt")
    assert np.allclose(np.array(refocused_cpu).flatten().view(float),
                       reference)


@skip_if_missing("cupy")
def test_refocus_cupy_stack_matches_per_slice(set_ndarray_backend_to_cupy):
    rng = np.random.default_rng(0)
    stack = (rng.normal(size=(5, 16, 12)) + 1j * rng.normal(size=(5, 16, 12))
             ).astype(np.complex128)

    pixel_size = 1e-6
    dist = 2.13 * pixel_size

    rf_stack = nrefocus.RefocusCupy(
        field=stack,
        wavelength=8.25 * pixel_size,
        pixel_size=pixel_size,
        medium_index=1.533,
        distance=0,
        kernel="helmholtz",
        padding=True,
    )
    out_stack = rf_stack.propagate(distance=dist).get()

    out_expected = np.empty_like(stack)
    for i in range(stack.shape[0]):
        rf = nrefocus.RefocusCupy(
            field=stack[i],
            wavelength=8.25 * pixel_size,
            pixel_size=pixel_size,
            medium_index=1.533,
            distance=0,
            kernel="helmholtz",
            padding=True,
        )
        out_expected[i] = rf.propagate(distance=dist).get()

    assert out_stack.shape == stack.shape
    assert np.allclose(out_stack, out_expected, rtol=1e-12, atol=1e-12)
