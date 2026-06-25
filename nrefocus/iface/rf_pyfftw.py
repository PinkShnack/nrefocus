import multiprocessing as mp

import pyfftw

from .. import pad

from .base import Refocus


class RefocusPyFFTW(Refocus):
    """Refocusing with FFTW

    .. versionadded:: 0.4.0

    """
    backend_expected = "numpy"
    # pyfftw can't used `cupy` ndarrays
    backend_incompatible = "cupy"

    def _init_fft(self, field, padding):
        """Perform initial Fourier transform of the input field

        Parameters
        ----------
        field: 2d complex-valued ndarray
            Input field to be refocused
        padding: bool
            Wheter to perform boundary-padding with linear ramp

        Returns
        -------
        fft_field0: 2d complex-valued ndarray
            Fourier transform the initial field

        Notes
        -----
        The number of threads in PyFFTW is currently set to the
        number of CPUs via `multiprocessing.cpu_count()`.
        """
        if padding:
            field = pad.pad_add(field)
        spatial_axes = (field.ndim - 2, field.ndim - 1)
        # compute the input Fourier transform
        origin = pyfftw.empty_aligned(field.shape, dtype='complex128')
        fft_origin = pyfftw.empty_aligned(field.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(origin, fft_origin, axes=spatial_axes)
        origin[:] = field
        fft_obj()

        # now setup the backward transform
        inv_input = pyfftw.empty_aligned(field.shape, dtype='complex128')
        inv_output = pyfftw.empty_aligned(field.shape, dtype='complex128')
        self._ifft_obj = pyfftw.FFTW(inv_input, inv_output, axes=spatial_axes,
                                     direction="FFTW_BACKWARD",
                                     flags=["FFTW_DESTROY_INPUT"],
                                     threads=mp.cpu_count())
        return fft_origin

    def propagate(self, distance):
        fft_kernel = self.get_kernel(distance=distance)
        fft_prop = self.fft_origin * fft_kernel
        if self.output_domain == "fourier":
            return pyfftw.interfaces.numpy_fft.fftshift(fft_prop)

        # `out=` does not permit broadcasting; explicitly assign the
        # broadcasted product into the input buffer.
        self._ifft_obj.input_array[:] = fft_prop
        refoc = self._ifft_obj()
        if self.input_domain == "spatial" and self.padding:
            refoc = pad.pad_rem(refoc)
        return refoc
