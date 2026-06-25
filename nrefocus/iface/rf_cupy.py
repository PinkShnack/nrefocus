import warnings
import scipy as sp
import cupyx.scipy.fft as cufft

from .._ndarray_backend import xp, NDArrayBackendWarning
from .. import pad
from .base import Refocus


class RefocusCupy(Refocus):
    """Refocusing with cupy-based Fourier transform

    .. versionadded:: 0.6.0

    """
    backend_expected = "cupy"
    backend_incompatible = None

    def _init_fft(self, field, padding):
        """Perform initial Fourier transform of the input field

        Parameters
        ----------
        field: 2d complex-valued ndarray
            Input field to be refocused
        padding: bool
            Whether to perform boundary-padding with linear ramp

        Returns
        -------
        fft_field0: 2d complex-valued ndarray
            Fourier transform the initial field
        """
        if not xp.is_cupy():
            warnings.warn(NDArrayBackendWarning(
                "You are using `RefocusCupy` without the 'cupy' ndarray "
                "backend. This will limit the Refocussing speed. "
                "To set the ndarray "
                "backend, use `nrefocus.set_ndarray_backend('cupy')` "))

        field_gpu = xp.asarray(field)
        if padding:
            field_gpu = pad.pad_add(field_gpu)
        with sp.fft.set_backend(cufft):
            # Allow stacks shaped (..., y, x) by transforming the last axes.
            return sp.fft.fft2(field_gpu, axes=(-2, -1))

    def propagate(self, distance):
        if not xp.is_cupy():
            warnings.warn(UserWarning(
                "You are using `RefocusCupy` without the 'cupy' ndarray "
                "backend. This will limit the Refocussing speed. "
                "To set the ndarray "
                "backend, use `nrefocus.set_ndarray_backend('cupy')` "))

        fft_kernel = self.get_kernel(distance=distance)
        fft_gpu = xp.asarray(self.fft_origin * fft_kernel)

        if self.output_domain == "fourier":
            return xp.fft.fftshift(fft_gpu, axes=(-2, -1))

        with sp.fft.set_backend(cufft):
            refoc = sp.fft.ifft2(fft_gpu, axes=(-2, -1))
        if self.input_domain == "spatial" and self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
