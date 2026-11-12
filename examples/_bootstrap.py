"""
Make `import fedbatch_simulator` work from a fresh clone.

The package modules live in the repository root, so the folder would have to
be named `fedbatch_simulator` and its parent be on sys.path. This registers
the repository root as the package `fedbatch_simulator` whatever the folder
is called. Importing this module first is all an example needs to do.
"""
import importlib.util
import pathlib
import sys

if 'fedbatch_simulator' not in sys.modules:
    _root = pathlib.Path(__file__).resolve().parents[1]
    _spec = importlib.util.spec_from_file_location(
        'fedbatch_simulator', _root / '__init__.py',
        submodule_search_locations=[str(_root)])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules['fedbatch_simulator'] = _module
    _spec.loader.exec_module(_module)
