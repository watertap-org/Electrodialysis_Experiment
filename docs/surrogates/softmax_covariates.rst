``softmax_covariates.py``
=========================

Location
--------

``src/electrodialysis_experiment/surrogates/transport_number_membrane/softmax_covariates.py``

Purpose
-------

Implements a softmax-based surrogate for cation transport numbers in CEM,
along with feature preparation, offline fitting, and standalone prediction
utilities.

Surrogate form
--------------

Registered builder:

- ``softmax_covariates``:
  uses a multinomial-logit-style score model with softmax normalization.

This form enforces by construction:

- strictly positive transport numbers for all modeled cations,
- unit-sum closure across cation transport numbers,
- explicit dependence on local composition and current density.

Feature construction
--------------------

The current implementation uses three covariates:

- ``log_ca_na``:
  :math:`\log((c_{Ca}+\epsilon)/(c_{Na}+\epsilon))`
- ``log_mg_na``:
  :math:`\log((c_{Mg}+\epsilon)/(c_{Na}+\epsilon))`
- ``log_current_density``:
  :math:`\log(j+\epsilon)`

These are prepared from tabular data by
``prepare_softmax_feature_data_from_dataframe(...)``.

When ``CurrD`` is not provided explicitly, current density is reconstructed as

.. math::

   j = \frac{I}{W_{\mathrm{eff}} L_{\mathrm{eff}}}

The initializer optionally standardizes these features using their empirical
mean and standard deviation across the fitting dataset.

Theoretical structure
---------------------

Let ``ref`` denote the reference cation, currently sodium in the main use
case, and let :math:`i \neq ref` denote non-reference cations.

For each non-reference ion, define a linear score

.. math::

   \eta_i = \alpha_i + \sum_{f} \beta_{i,f} z_f

where :math:`z_f` are the standardized feature values.

The transport numbers are then computed by softmax normalization:

.. math::

   t_{ref} = \frac{1}{1 + \sum_{i \neq ref} \exp(\eta_i)}

and

.. math::

   t_i = \frac{\exp(\eta_i)}{1 + \sum_{j \neq ref} \exp(\eta_j)}

This construction is inspired by multinomial-logit or softmax regression for
compositional responses. In this application, the response components are
cation transport numbers rather than class probabilities. The form is useful
because it preserves positivity and closure automatically, while still
allowing the relative transport preference among ions to vary with local state.

Role of the construction
------------------------

Compared with concentration-ratio-only surrogates, the softmax form is meant
to provide a more flexible description of state-dependent selectivity.

Its design reflects three modeling goals:

- composition sensitivity:
  transport preference should change with local :math:`Ca/Na` and
  :math:`Mg/Na` ratios,
- operating-intensity sensitivity:
  transport preference should also change with current density,
- physically admissible outputs:
  predicted transport numbers should remain positive and sum to one without
  needing a separate closure equation for non-reference ions.

Offline fitting routine
-----------------------

``init_softmax_covariates(...)``:

- requires ``feature_data`` and measured transport-number data,
- optionally standardizes the features,
- packs intercept and coefficient variables into one parameter vector,
- minimizes summed squared error across ions using SciPy ``L-BFGS-B``,
- returns fitted intercepts, coefficients, and feature-standardization data.

The returned coefficient dictionary contains:

- ``intercept``
- ``coef``
- ``feature_center``
- ``feature_scale``
- ``standardize_features``

Embedded Pyomo form
-------------------

When attached to the experiment model, the surrogate creates:

- feature expressions over membrane position ``x``,
- score expressions ``eta[ion, x]``,
- a softmax denominator expression,
- constraints equating membrane transport numbers to softmax outputs.

This allows the surrogate to respond to local ED state variables during the
solve, rather than only providing a static post-processing correlation.

Prediction utilities
--------------------

``softmax_covariates_fn(...)`` and
``predict_ti_by_softmax_covariates(...)`` provide forward evaluation of fitted
softmax coefficients outside the Pyomo model.

These utilities:

- apply the stored feature standardization,
- evaluate the softmax transport-number prediction,
- return predicted transport numbers for each ion.

Numerical tradeoffs
-------------------

- the softmax form gives physically admissible outputs automatically,
- feature standardization improves coefficient scaling and optimizer behavior,
- the use of exponentials can cause overflow if inferred scores become too
  large in extrapolative regimes,
- prediction quality depends on how well the chosen covariates span the
  operating-state dependence of membrane selectivity,
- as with other flexible surrogates, poor extrapolation can appear first in
  low-concentration or strongly deionized conditions where the inferred local
  state moves outside the training envelope.
