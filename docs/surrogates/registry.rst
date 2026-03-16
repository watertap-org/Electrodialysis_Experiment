``registry.py``
===============

Location
--------

``src/electrodialysis_experiment/surrogates/transport_number_membrane/registry.py``

Purpose
-------

Provides decorator-based registries for surrogate builders, initialization
functions, and optional identity/prediction functions.

Registries
----------

- ``SURROGATE_REGISTRY``:
  maps surrogate name -> Pyomo block builder.
- ``SURROGATE_INITIALIZE``:
  maps surrogate name -> offline fit initializer.
- ``SURROGATE_IDENTITY``:
  maps surrogate name -> standalone prediction function.

Decorators
----------

``register_transport_number_surrogate(name)``
  Register block-construction function.

``register_transport_number_surrogate_init(name)``
  Register offline fitting initializer.

``register_transport_number_surrogate_fn_identity(name)``
  Register functional identity/prediction callable.

Role in training stack
----------------------

This module decouples surrogate dispatch from hardcoded conditionals. The
simulator block can select a surrogate by name, and both equation build and
initialization logic are resolved dynamically from these registries.

Practical note
--------------

Duplicate surrogate keys raise ``KeyError`` at import/registration time.
