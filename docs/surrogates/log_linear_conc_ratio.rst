``log_linear_conc_ratio.py``
============================

Location
--------

``src/electrodialysis_experiment/surrogates/transport_number_membrane/log_linear_conc_ratio.py``

Purpose
-------

Implements the surrogate equations and offline fitting routines for
concentration-ratio-based cation transport-number models.

Surrogate forms
---------------

Registered builders:

- ``log_linear_polynomial``:
  uses Taylor approximation of ``log(1 + A)`` up to chosen degree.
- ``log_linear_log``:
  uses exact ``log(1 + A)`` form.

Both forms enforce:

- non-reference cation equations in log space,
- reference-ion closure so cation transport numbers sum to one.

Theoretical structure
---------------------

For non-reference ion ``i``:

.. math::

   \log(t_i + \epsilon)
   = \log(\beta_i + \epsilon)
   + \log\left(\frac{c_i + \epsilon}{c_{ref} + \epsilon}\right)
   - \log(1 + A)

with

.. math::

   A = \sum_{j \neq ref} \beta_j
   \frac{c_j + \epsilon}{c_{ref} + \epsilon}

and closure:

.. math::

   t_{ref} = 1 - \sum_{j \neq ref} t_j

``\epsilon`` is a small numerical regularizer to keep log/ratio terms defined
near zero and stabilize derivatives for NLP solve.

Offline fitting routine
-----------------------

``init_log_linear_polynomial(...)``:

- builds concentration-ratio arrays from experimental inputs,
- defines SSE objective (linear or log-residual variant),
- solves coefficient regression with multiple SciPy optimizers and initial
  guesses,
- reports fit quality (combined and per-ion :math:`R^2`),
- returns fitted coefficient dictionary.

Prediction utilities
--------------------

``log_linear_polynomial_fn(...)`` and ``predict_ti_by_surrogate(...)`` provide
forward evaluation of fitted surrogate coefficients outside the Pyomo model.

Numerical tradeoffs
-------------------

- polynomial approximation can improve smoothness/control but introduces
  truncation error for large ``A``,
- exact-log form avoids Taylor truncation but can be harder numerically in some
  regimes,
- coefficient bounds and initialization strongly influence fit robustness.
