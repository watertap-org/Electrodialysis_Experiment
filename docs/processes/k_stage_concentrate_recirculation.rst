``k_stage_concentrate_recirculation.py``
========================================

Location
--------

``src/electrodialysis_experiment/processes/k_stage_concentrate_recirculation.py``

Purpose
-------

``k_stage_concentrate_recirculation.py`` extends the staged single-pass
process wrapper to a **k-stage feed-and-bleed electrodialysis train** with one
concentrate recycle loop wrapped around the full stack train.

The central class is ``KStageConcentrateRecirculation`` (implemented as
``KStageConcentrateRecirculationData`` through
``declare_process_block_class``).

High-level process structure
----------------------------

This module combines two ideas already present elsewhere in the repository:

- the stage-indexed ED train from ``k_stage_single_pass.py``, and
- the recycle-loop topology from
  ``one_stage_concentrate_recirculation.py``.

The resulting flowsheet has:

- one shared ``Feed`` block,
- one upstream ``Separator`` ``sepa0``,
- one recycle ``Mixer`` ``mix0``,
- two shared pumps,
- one ``ED_base`` unit per stage under ``fs.EDstack[stage].unit``,
- one downstream ``Separator`` ``sepa1`` for bleed/recycle splitting,
- one product outlet block, and
- one disposal outlet block.

The defining recycle path is:

``final-stage concentrate outlet -> sepa1 -> mix0 -> pump0 -> stage-1 concentrate inlet``

How it differs from ``KStageSinglePass``
----------------------------------------

``KStageSinglePass`` has one pass through the stage train and no recycle loop.
``KStageConcentrateRecirculation`` adds:

- a concentrate-side recycle mixer,
- a downstream bleed/recycle separator,
- a configurable overall recycle fraction,
- a recoverable ``fs.recovery_vol_H2O`` variable with explicit constraint, and
- recycle-aware initialization logic.

This makes it suitable for feed-and-bleed operation where users may want to
control water recovery while still concentrating the reject stream through
recirculation.

Configuration model
-------------------

The wrapper uses ``KStageSinglePassConfig`` and the same stage-aware YAML
pattern as ``k_stage_single_pass.py``:

- default stack block via ``ed_stack``,
- stage-specific overrides via ``ed_stacks``,
- ``num_stages`` to set the length of the train.

The usual entry point is ``KStageConcentrateRecirculation.from_yaml(...)``.

YAML files commonly used with this module
-----------------------------------------

Typical inputs are:

- process config:
  ``src/electrodialysis_experiment/configs/two_stage_conc_recir.yml``
- initialization values:
  ``src/electrodialysis_experiment/configs/tscr_init_config.yml``
- scaling factors:
  ``src/electrodialysis_experiment/configs/scaling_ts_cc.yml``

For YAML structure details, see :doc:`../utils/yaml_configuration_guide` and
:doc:`../schema/process_config_schema`.

Important public methods
------------------------

``from_yaml(path)``
  Build the staged recirculation wrapper from YAML.

``import_init_value_config(path)``
  Apply stage-aware initialization and fixed values.

``import_scaling_config(path)``
  Apply stage-aware scaling factors.

``initialize_process(...)``
  Initialize feed, separators, pumps, stage train, recycle split, disposal,
  and recycle mixer, then optionally solve the coupled flowsheet.

``set_feed_split_to_diluate(split_fraction, fix=True)``
  Set or fix the upstream feed split between the diluate path and fresh
  concentrate path.

``set_concentrate_recycle_fraction(recycle_fraction, fix=True)``
  Set or fix the downstream recycle fraction.

``display_selected_model_metrics(...)``
  Print feed/product/disposal values, stage-wise ion treatment values,
  stage-wise performance metrics, pressure/temperature checks, and OCV values.

Recovery and recycle closure
----------------------------

As in the one-stage recycle module, ``fs.recovery_vol_H2O`` is intentionally
implemented as a ``Var`` with an explicit constraint rather than an
``Expression``. That allows the process to be closed either by:

- fixing water recovery, or
- fixing recycle split fraction.

The module keeps the same practical initialization behavior:

- if the recycle split is already fixed by the caller, initialization respects
  that choice,
- otherwise a temporary recycle split is fixed to initialize ``sepa1``, then
  restored to free status afterward.

Initialization sequence
-----------------------

The process is initialized in a staged and recycle-aware order:

1. Feed is initialized from the provided ``FluidCondition``.
2. ``sepa0`` is initialized.
3. Shared pumps are initialized.
4. Stage 1 is initialized.
5. Inter-stage states are propagated and each downstream stage is initialized.
6. Product block is initialized from final-stage diluate outlet.
7. ``sepa1`` is initialized from final-stage concentrate outlet.
8. Disposal block is initialized.
9. Recycle state is propagated back to the mixer.
10. Mixer is initialized.
11. The full flowsheet is optionally solved.

This sequence reduces cold-start difficulty while preserving the coupled
multi-stage recycle topology.

When to use this module
-----------------------

Use ``KStageConcentrateRecirculation`` when:

- more than one ED stage is needed,
- a final concentrate recycle loop must be represented,
- feed-and-bleed operation is the intended process concept, or
- stage-wise current, OCV, and concentration evolution matter.

When not to use this module
---------------------------

If only one stage is needed, ``one_stage_concentrate_recirculation.py`` is
simpler. If there is no recycle loop, ``k_stage_single_pass.py`` is the more
appropriate process wrapper.
