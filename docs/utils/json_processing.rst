``json_processing.py``
======================

Location
--------

``src/electrodialysis_experiment/utils/json_processing.py``

Purpose
-------

Converts nested JSON prediction records into element-wise NumPy arrays suitable
for downstream metric or plotting workflows.

Core function
-------------

``predicted_values_to_arrays(data_or_path, ...)``

Input shape expected per record:

- sample identifier key (default ``ti_param_sample_id``),
- ``predicted_values`` dict keyed by case names (for example ``case_1``,
  ``case_2``),
- each case containing element/value pairs.

Return values:

- ``arrays_by_element``:
  ``dict[element] -> ndarray(n_samples, n_cases)``,
- ``sample_ids``:
  row order,
- ``case_names``:
  column order (numerically sorted for ``case_<n>`` patterns).

Practical notes
---------------

- Missing element/case values are filled with ``fill_value`` (default ``NaN``).
- You can pass file path or preloaded Python list of dicts.
- ``element_keys`` can be forced to keep a fixed output schema.
