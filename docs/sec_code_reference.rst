==============
Code reference
==============

.. _sec_refocus_interface:

Refocus interface
=================
`Refocus` is a user-convenient interface for numerical refocusing.
Each class implements refocusing for a specific dimensionality (1D or
2D fields) using a specific method for refocusing (e.g. numpy FFT or FFTW).
For 2D interfaces, inputs can be ``(y, x)`` or ``(..., y, x)``; when users
pass an ``n``-dimensional input, the output array keeps that same
``n``-dimensional shape.

.. autofunction:: nrefocus.get_best_interface

.. autoclass:: nrefocus.RefocusPyFFTW
    :members:
    :inherited-members:

.. autoclass:: nrefocus.RefocusNumpy
    :members:
    :inherited-members:

.. autoclass:: nrefocus.RefocusNumpy1D
    :members:
    :inherited-members:



Metrics
=======
.. automodule:: nrefocus.metrics
   :members:
   :imported-members:


Minimizers
==========
.. automodule:: nrefocus.minimizers
   :members:
   :imported-members:


Legacy methods
==============
These methods are legacy functions which are kept for backwards-compatibility.

Refocusing
----------
.. autofunction:: nrefocus.refocus

.. autofunction:: nrefocus.refocus_stack


Autofocusing
------------
.. autofunction:: nrefocus.autofocus

.. autofunction:: nrefocus.autofocus_stack

