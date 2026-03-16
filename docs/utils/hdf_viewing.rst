``hdf_viewing.py``
==================

Location
--------

``src/electrodialysis_experiment/utils/hdf_viewing.py``

Purpose
-------

Utility helpers for inspecting model snapshots stored by
``MasterExperimentBuilder.save_model_hdf(...)``.

Core API
--------

``list_hdf_keys(h5_path)``
  List HDF tables present in a snapshot file.

``load_mod_hdf(h5_path, key)``
  Load one table (variables / parameters / constraints / expressions) as a
  DataFrame.

``search_components(h5_path, kind, comp_keywords, index_keywords=None, ...)``
  Filter rows by AND-matched component/index keywords and return structured
  search output.

``SearchResult``
  Dataclass container with:

- ``matches``: matched rows,
- ``by_component``: grouped rows by component path,
- ``case``: compact match classification (0=no match, 1=single row, 2=single
  component with multiple indices, 3=multiple components).

How this fits the workflow
--------------------------

After calibration runs, this module is the fastest way to inspect what was
stored in snapshots, confirm specific variables were restored, and compare
component-level state across runs.
