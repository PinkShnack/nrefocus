"""Refocus a stack of 2D fields (3D array input).

This example shows how to refocus a stack shaped (n, y, x).  nrefocus
interprets the last two axes as spatial axes and broadcasts over any leading
"batch" axes.

"""
import numpy as np

import nrefocus

rng = np.random.default_rng(0)

# Synthetic complex field stack: (n, y, x)
n, ny, nx = 5, 128, 96
amp = 1 + 0.1 * rng.normal(size=(n, ny, nx))
pha = 0.3 * rng.normal(size=(n, ny, nx))
field = amp * np.exp(1j * pha)

pixel_size = 1e-6
dist = 2.13 * pixel_size

# CPU (NumPy) refocus of the entire stack in one call
rf = nrefocus.RefocusNumpy(
    field=field,
    wavelength=8.25 * pixel_size,
    pixel_size=pixel_size,
    medium_index=1.533,
    distance=0,
    kernel="helmholtz",
    padding=True,
)
refocused = rf.propagate(distance=dist)
print("nrefocus now accepts stacks of 2D arrays (3D input)!\n"
      "Input shape:", field.shape, "| Output shape-> ", refocused.shape)
