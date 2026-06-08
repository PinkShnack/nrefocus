"""Specifying `input_domain` / `output_domain` in nrefocus.

As of version 0.6.2, users can specify in which domain the input and output
data exists. The options are `'spatial'` (default, legacy behaviour) and
`'fourier'`. These can be used in conjunction with
`qpretrieve`'s `'output_domain'` to reduce the number of Fourier
transforms in a given pipeline (see https://qpretrieve.readthedocs.io/).

Each of the three subplots below produce the same spatial result when padding=False.
Note that `qpretrieve` is not used in this example.

"""
import matplotlib.pyplot as plt
import numpy as np
import nrefocus

from example_helper import load_cell

field = load_cell("HL60_field.zip")
fft_field = np.fft.fftshift(np.fft.fft2(field))

propagation_kwargs = dict(d=15, nm=1, res=1, method="helmholtz", padding=False)

# 1. default (spatial in, spatial out)
refocused_spatial = nrefocus.refocus(field=field, **propagation_kwargs)

# 2. fourier input (skip forward FFT)
refocused_fourier_in = nrefocus.refocus(
    field=fft_field,
    input_domain="fourier",
    **propagation_kwargs,
)

# 3. fourier output, then manual iFFT
refocused_fft_out = nrefocus.refocus(
    field=field,
    output_domain="fourier",
    **propagation_kwargs,
)
# if the field came from qpretrieve, we could use the new feature
# `oah.compute_field(refocused_fft_out)`
refocused_from_fft_out = np.fft.ifft2(np.fft.ifftshift(refocused_fft_out))

assert np.allclose(refocused_spatial, refocused_fourier_in, atol=1e-12), (
    "1 and 2 differ.")
assert np.allclose(refocused_spatial, refocused_from_fft_out, atol=1e-12), (
    "1 and 3 differ after iFFT.")
print("All outputs match:", True)

# plot the output fields
amp = [
    np.abs(refocused_spatial),
    np.abs(refocused_fourier_in),
    np.abs(refocused_from_fft_out),
]
titles = [
    "(default)\ninput_domain='spatial'\noutput_domain='spatial'",
    "\ninput_domain='fourier'\noutput_domain='spatial'",
    "(+ manual iFFT)\ninput_domain='fourier'\noutput_domain='spatial'",
]

vmin = min(a.min() for a in amp)
vmax = max(a.max() for a in amp)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
for ax, a, title in zip(axes, amp, titles):
    im = ax.imshow(a, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].set_ylabel("y (px)", fontsize=7)

fig.colorbar(im, ax=axes, label="amplitude (a.u.)", shrink=0.8)
plt.savefig("input_output_domains.png", dpi=150, bbox_inches="tight")
plt.show()
