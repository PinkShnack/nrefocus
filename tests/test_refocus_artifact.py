import numpy as np

import nrefocus


class DummyArtifact:
    """Dummy class that shows the basic idea of the
    :func:`qpretrieve.fourier.FourierFieldArtifact.finalize` class."""
    output_domain = "spatial"

    def __init__(self, field):
        self.fft_used = np.fft.fftshift(np.fft.fft2(field))
        self.last_propagated_fft = None

    def finalize(self, propagated_fft=None):
        self.last_propagated_fft = propagated_fft
        return np.fft.ifft2(np.fft.ifftshift(propagated_fft))


def test_refocus_artifact_fourier_output_returns_shifted_fft(dummy_field):
    """Artifact and output_domain='fourier' gives fftshifted propagated FFT."""
    artifact = DummyArtifact(dummy_field)
    kwargs = dict(d=2.13, nm=1.533, res=8.25,
                  method="helmholtz", padding=False)

    refocused_spatial = nrefocus.refocus(field=dummy_field, **kwargs)

    # artifact fourier output should be fftshift(fft2(refocused_spatial))
    fft_out = nrefocus.refocus(
        field=artifact, output_domain="fourier", **kwargs)
    refocused_from_fft = np.fft.ifft2(np.fft.ifftshift(fft_out))

    # finalize() is not called for fourier output, fft_out is returned directly
    assert artifact.last_propagated_fft is None
    assert np.allclose(refocused_from_fft, refocused_spatial,
                       rtol=1e-12, atol=1e-12)


def test_refocus_artifact_matches_default_field_path(dummy_field):
    """Old and new pipelines should give the same results"""
    artifact = DummyArtifact(dummy_field)
    kwargs = dict(d=2.13, nm=1.533, res=8.25,
                  method="helmholtz", padding=False)

    refocused_field = nrefocus.refocus(field=dummy_field, **kwargs)
    refocused_artifact = nrefocus.refocus(field=artifact, **kwargs)

    assert artifact.last_propagated_fft is not None
    assert np.allclose(refocused_artifact, refocused_field,
                       rtol=1e-12, atol=1e-12)


def test_refocus_artifact_matches_default_field_real_data(cell_field):
    """Old and new pipelines should give the same results"""
    artifact = DummyArtifact(cell_field)
    kwargs = dict(d=2.13, nm=1.533, res=8.25,
                  method="helmholtz", padding=False)

    refocused_field = nrefocus.refocus(field=cell_field, **kwargs)
    refocused_artifact = nrefocus.refocus(field=artifact, **kwargs)

    assert artifact.last_propagated_fft is not None
    assert np.allclose(refocused_artifact, refocused_field,
                       rtol=1e-12, atol=1e-12)


def test_refocus_fourier_input_domain_matches_spatial_path(dummy_field):
    """Show that default spatial pipeline matches when input is fourier"""
    fft_field = np.fft.fftshift(np.fft.fft2(dummy_field))
    kwargs = dict(d=1.5, nm=1.533, res=8.25,
                  method="fresnel", padding=False)

    refocused_spatial = nrefocus.refocus(field=dummy_field, **kwargs)
    refocused_fourier = nrefocus.refocus(
        field=fft_field, input_domain="fourier", **kwargs)

    assert np.allclose(refocused_fourier, refocused_spatial,
                       rtol=1e-12, atol=1e-12)


def test_refocus_fourier_input_domain_ignores_padding(dummy_field):
    """Show that when input is fourier, the padding is ignored"""
    fft_field = np.fft.fftshift(np.fft.fft2(dummy_field))
    kwargs = dict(d=0.75, nm=1.533, res=8.25,
                  method="helmholtz", padding=True)

    refocused_fourier_padding = nrefocus.refocus(
        field=fft_field, input_domain="fourier", **kwargs)
    refocused_fourier_no_padding = nrefocus.refocus(
        field=fft_field, input_domain="fourier", **dict(kwargs, padding=False))

    assert np.allclose(refocused_fourier_padding, refocused_fourier_no_padding,
                       rtol=1e-12, atol=1e-12)
    assert refocused_fourier_padding.shape == fft_field.shape


def test_refocus_fourier_output_domain_returns_fft(dummy_field):
    """The fourier output domain should return an FFT"""
    kwargs = dict(d=0.5, nm=1.533, res=8.25, method="helmholtz", padding=False,
                  input_domain="spatial", output_domain="fourier")

    refocused_fft = nrefocus.refocus(field=dummy_field, **kwargs)
    rf = nrefocus.RefocusNumpy(
        field=dummy_field, wavelength=kwargs["res"] * 1e-6, pixel_size=1e-6,
        medium_index=kwargs["nm"], distance=0, kernel=kwargs["method"],
        padding=kwargs["padding"], input_domain=kwargs["input_domain"],
        output_domain=kwargs["output_domain"])
    expected_fft = np.fft.fftshift(
        rf.fft_origin * rf.get_kernel(distance=kwargs["d"] * 1e-6))

    assert np.allclose(refocused_fft, expected_fft, rtol=1e-12, atol=1e-12)
    assert rf.output_domain == "fourier"


def test_refocus_fourier_input_and_output_domains_return_fft(dummy_field):
    """The fourier input and output domain should return an FFT"""
    fft_field = np.fft.fft2(dummy_field)
    kwargs = dict(d=1.25, nm=1.533, res=8.25, method="fresnel", padding=False,
                  input_domain="fourier", output_domain="fourier")

    refocused_fft = nrefocus.refocus(field=fft_field, **kwargs)
    rf = nrefocus.RefocusNumpy(
        field=fft_field, wavelength=kwargs["res"] * 1e-6, pixel_size=1e-6,
        medium_index=kwargs["nm"], distance=0, kernel=kwargs["method"],
        padding=kwargs["padding"], input_domain=kwargs["input_domain"],
        output_domain=kwargs["output_domain"])
    expected_fft = np.fft.fftshift(
        rf.fft_origin * rf.get_kernel(distance=kwargs["d"] * 1e-6))

    assert np.allclose(refocused_fft, expected_fft, rtol=1e-12, atol=1e-12)
    assert rf.input_domain == "fourier"
    assert rf.output_domain == "fourier"
