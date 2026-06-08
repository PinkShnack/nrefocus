import numpy as np
import nrefocus

_rng = np.random.default_rng(0)
_FIELD = _rng.normal(size=(256, 256)) + 1j * _rng.normal(size=(256, 256))
_FFT_FIELD = np.fft.fftshift(np.fft.fft2(_FIELD))

_KWARGS = dict(d=1.5, nm=1.333, res=8.25, method="helmholtz", padding=False)


def test_bm_refocus_spatial_default(benchmark):
    """Default spatial input and spatial output."""
    result = benchmark(nrefocus.refocus, field=_FIELD, **_KWARGS)
    assert result.shape == _FIELD.shape


def test_bm_refocus_fourier_input_spatial_output(benchmark):
    """Fourier input skips the forward FFT."""
    result = benchmark(nrefocus.refocus, field=_FFT_FIELD,
                       input_domain="fourier", **_KWARGS)
    assert result.shape == _FIELD.shape


def test_bm_refocus_spatial_input_fourier_output(benchmark):
    """Fourier output skips the inverse FFT."""
    result = benchmark(nrefocus.refocus, field=_FIELD,
                       output_domain="fourier", **_KWARGS)
    assert result.shape == _FIELD.shape


def test_bm_refocus_fourier_input_fourier_output(benchmark):
    """Fourier as both input and output does no transforms."""
    result = benchmark(
        nrefocus.refocus, field=_FFT_FIELD, input_domain="fourier",
        output_domain="fourier", **_KWARGS)
    assert result.shape == _FIELD.shape
