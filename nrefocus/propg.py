import multiprocessing as mp
from ._ndarray_backend import xp

from . import iface


__all__ = ["refocus", "refocus_stack"]


_cpu_count = mp.cpu_count()


def _is_fourier_field_artifact(field) -> bool:
    return hasattr(field, "fft_used") and hasattr(field, "finalize")


def refocus(field, d, nm, res, method="helmholtz", padding=True,
            input_domain="spatial", output_domain="spatial"):
    """Refocus a 1D field or a 2D field / stack of 2D fields

    Parameters
    ----------
    field : 1d array or (..., y, x) array
        Background corrected electric field (Ex/BEx). For stacks, the last
        two axes are interpreted as spatial axes.
        The interpretation of `field` is controlled by `input_domain`.
        An object with `fft_used` and `finalize()` is also accepted; in that
        case the precomputed Fourier field is propagated directly.
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(idk²/kₘ)`
    padding : bool
        perform padding with linear ramp from edge to average
        to reduce ringing artifacts.
        Ignored for objects with `fft_used` and `finalize()`, which is the
        case of the qpretrieve `artifact` object.

         .. versionadded:: 0.1.4
    input_domain : str
        Either ``"spatial"`` or ``"fourier"``. Default is ``"spatial"`` and
        treats `field` as in older nrefocus versions.
        If ``"fourier"``, `field` is treated as a precomputed Fourier-domain
        field and the initial FFT step is skipped.

        .. versionadded:: 0.8.0
    output_domain : str
        Either ``"spatial"`` or ``"fourier"``. Default is ``"spatial"`` and
        outputs inverse transformed `field` as in older nrefocus versions.
        If ``"fourier"``, the propagated Fourier-domain field is
        returned directly.

        .. versionadded:: 0.8.0

    Returns
    -------
    Returns the propagated field in the requested `output_domain`.

    Notes
    -----
    This method uses :class:`nrefocus.RefocusNumpy` for refocusing
    of 2D fields. This is because the :func:`nrefocus.refocus_stack`
    function uses `async` which appears to not work with e.g.
    :mod:`pyfftw`. Use `rf = nrefocus.iface.RefocusCupy` or `RefocusPyFFTW`
    syntax if you want to use CuPy or PyFFTW.

    """
    if _is_fourier_field_artifact(field):
        # go straight to propagation if `field` is in Fourier domain
        return _refocus_artifact(field, d, nm, res, method, output_domain)

    if input_domain not in ("spatial", "fourier"):
        raise ValueError("`input_domain` must be 'spatial' or 'fourier'.")

    fshape = len(field.shape)
    if fshape == 1:
        # 1D field
        rfcls = iface.RefocusNumpy1D
    elif fshape >= 2:
        # 2D field or stack (..., y, x)
        rfcls = iface.RefocusNumpy
    else:
        raise AssertionError(f"Unexpected dimension of `field` ({fshape}).")

    # use a made-up pixel size so we can use the new `Refocus` interface
    pixel_size = 1e-6
    rf = rfcls(field=field,
               wavelength=res*pixel_size,
               pixel_size=pixel_size,
               medium_index=nm,
               distance=0,
               kernel=method,
               padding=padding,
               input_domain=input_domain,
               output_domain=output_domain
               )
    refoc = rf.propagate(distance=d*pixel_size)

    return refoc


def _refocus_artifact(field, d, nm, res, method="helmholtz",
                      output_domain="spatial"):
    """Refocus a precomputed Fourier-domain field artifact.

    The artifact path always starts from `field.fft_used`. Fourier output is
    returned in qpretrieve's shifted convention. Spatial output delegates the
    final crop/scale step back to qpretrieve via ``field.finalize(...)``.

    padding is always False, the assumption is that the user has a padded
    or square Fourier transform as input e.g., if using `qpretrieve`, the
    padding is already baked into `fft_used`, so nrefocus must not pad again.

    """
    fshape = len(field.fft_used.shape)
    if fshape == 1:
        rfcls = iface.RefocusNumpy1D
    elif fshape >= 2:
        rfcls = iface.RefocusNumpy
    else:
        raise AssertionError(
            f"Unexpected dimension of `fft_used` ({fshape}).")

    pixel_size = 1e-6
    # field.fft_used is in fftshifted convention (DC/sideband at centre),
    # as stored by qpretrieve after fftshift(fft2(data)).
    # input_domain="fourier" tells Refocus.__init__ to apply ifftshift so
    # that fft_origin is in the unshifted layout required by fftfreq kernels.
    rf = rfcls(field=field.fft_used,
               wavelength=res*pixel_size,
               pixel_size=pixel_size,
               medium_index=nm,
               distance=0,
               kernel=method,
               padding=False,
               input_domain="fourier")
    fft_kernel = rf.get_kernel(distance=d*pixel_size)
    # fft_origin and fft_kernel are both in unshifted layout.
    propagated_fft = rf.fft_origin * fft_kernel
    # fftshift returns to the fftshifted convention expected by
    # field.finalize(), which applies ifftshift before ifft2 internally.
    shifted_fft = xp.fft.fftshift(propagated_fft, axes=(-2, -1))
    if output_domain == "fourier":
        # Return fftshifted FFT directly; caller is responsible for ifftshift
        # + ifft2 if a spatial field is needed.
        return shifted_fft
    return field.finalize(shifted_fft)


def refocus_stack(fieldstack, d, nm, res, method="helmholtz",
                  num_cpus=_cpu_count, copy=True, padding=True,
                  input_domain="spatial", output_domain="spatial"):
    """Refocus a stack of 1D or 2D fields


    Parameters
    ----------
    fieldstack : 2d or 3d array
        Stack of 1D or 2D background corrected electric fields (Ex/BEx).
        The first axis iterates through the individual fields.
    d : float
        Distance to be propagated in pixels (negative for backwards)
    nm : float
        Refractive index of medium
    res : float
        Wavelength in pixels
    method : str
        Defines the method of propagation;
        one of

            - "helmholtz" : the optical transfer function `exp(idkₘ(M-1))`
            - "fresnel"   : paraxial approximation `exp(idk²/kₘ)`

    num_cpus : int
        Defines the number of CPUs to be used for refocusing.
    copy : bool
        If False, overwrites input stack.
    padding : bool
        Perform padding with linear ramp from edge to average
        to reduce ringing artifacts.

        .. versionadded:: 0.1.4
    input_domain : str
        Either ``"spatial"`` or ``"fourier"``. Passed through to
        :func:`refocus`.

        .. versionadded:: 0.8.0
    output_domain : str
        Either ``"spatial"`` or ``"fourier"``. Passed through to
        :func:`refocus`.

        .. versionadded:: 0.8.0

    Returns
    -------
    Returns the propagated stack in the requested `output_domain`.
    """
    func = refocus
    names = func.__code__.co_varnames[:func.__code__.co_argcount]

    loc = locals()
    vardict = dict()
    for name in names:
        if name in loc.keys():
            vardict[name] = loc[name]
    # default keyword arguments
    func_def = func.__defaults__[::-1]

    vardict["padding"] = padding
    vardict["input_domain"] = input_domain
    vardict["output_domain"] = output_domain

    M = fieldstack.shape[0]
    stackargs = list()

    # Create individual arglists for all fields
    for m in range(M):
        kwarg = vardict.copy()
        kwarg["field"] = fieldstack[m]
        # now we turn the kwarg into an arglist
        args = list()
        for i, a in enumerate(names[::-1]):
            # first set default
            if i < len(func_def):
                val = func_def[i]
            if a in kwarg:
                val = kwarg[a]
            args.append(val)
        stackargs.append(args[::-1])

    p = mp.Pool(num_cpus)
    result = p.map_async(_refocus_wrapper, stackargs).get()
    p.close()
    p.terminate()
    p.join()

    if copy:
        data = xp.zeros(fieldstack.shape, dtype=result[0].dtype)
    else:
        data = fieldstack

    for m in range(M):
        data[m] = result[m]

    return data


def _refocus_wrapper(args):
    """Just calls autofocus with *args. Needed for multiprocessing pool.
    """
    return refocus(*args)
