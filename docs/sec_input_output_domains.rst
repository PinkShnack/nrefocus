.. _sec_input_output_domains:

========================
Input and Output Domains
========================

Since version 0.7.1, :func:`nrefocus.refocus`, :func:`nrefocus.refocus_stack`,
and all :class:`.Refocus` subclasses accept ``input_domain`` and
``output_domain`` keyword arguments that control whether the field is
transformed to/from the Fourier domain inside nrefocus or whether the user
manages those transforms themselves.

This is useful when the field is already in Fourier space (e.g. it comes
directly from a hologram reconstruction library) or when only the propagated
Fourier coefficients are needed (e.g. as input to a further processing step),
avoiding redundant transforms.

.. note::

    When ``input_domain="fourier"``, the field must be provided in the
    **fftshifted** convention (DC at the centre of the array, as returned by
    :func:`numpy.fft.fftshift`).  When ``output_domain="fourier"``, the result
    is also returned fftshifted.

Default: spatial input, spatial output
---------------------------------------

The default behaviour applies a forward FFT to the input, multiplies by the
propagation kernel, then applies the inverse FFT.

.. code-block:: python

    import numpy as np
    import nrefocus

    ny, nx = 128, 96
    field = (np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
             ).astype(np.complex128)

    pixel_size = 1e-6
    refocused = nrefocus.refocus(
        field=field,
        d=2.13,
        nm=1.533,
        res=8.25,
        method="helmholtz",
        padding=False,
    )
    print(refocused.shape)  # (128, 96)

Fourier input: skip the forward FFT
-------------------------------------

If the field is already available as an fftshifted FFT, pass it with
``input_domain="fourier"`` to skip the forward transform inside nrefocus.

.. code-block:: python

    import numpy as np
    import nrefocus

    ny, nx = 128, 96
    field = (np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
             ).astype(np.complex128)

    # pre-compute the fftshifted FFT once
    fft_field = np.fft.fftshift(np.fft.fft2(field))

    pixel_size = 1e-6
    refocused = nrefocus.refocus(
        field=fft_field,
        d=2.13,
        nm=1.533,
        res=8.25,
        method="helmholtz",
        padding=False,
        input_domain="fourier",
    )
    print(refocused.shape)  # (128, 96)

Fourier output: skip the inverse FFT
--------------------------------------

If only the propagated Fourier coefficients are needed, use
``output_domain="fourier"`` to skip the inverse transform.  The result is
returned fftshifted; apply :func:`numpy.fft.ifft2` after
:func:`numpy.fft.ifftshift` to recover the spatial field.

.. code-block:: python

    import numpy as np
    import nrefocus

    ny, nx = 128, 96
    field = (np.random.randn(ny, nx) + 1j * np.random.randn(ny, nx)
             ).astype(np.complex128)

    pixel_size = 1e-6
    propagated_fft = nrefocus.refocus(
        field=field,
        d=2.13,
        nm=1.533,
        res=8.25,
        method="helmholtz",
        padding=False,
        output_domain="fourier",
    )
    # propagated_fft is fftshifted; recover the spatial field manually
    refocused = np.fft.ifft2(np.fft.ifftshift(propagated_fft))
    print(refocused.shape)  # (128, 96)

Combining both: no transforms at all
--------------------------------------

Setting both ``input_domain="fourier"`` and ``output_domain="fourier"``
reduces the pipeline to a single kernel multiplication — no FFT or iFFT is
performed inside nrefocus.

.. code-block:: python

    propagated_fft = nrefocus.refocus(
        field=fft_field,          # already fftshifted
        d=2.13,
        nm=1.533,
        res=8.25,
        method="helmholtz",
        padding=False,
        input_domain="fourier",
        output_domain="fourier",
    )

For a complete working example see the :ref:`sec_examples`.
