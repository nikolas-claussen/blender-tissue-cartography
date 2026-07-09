"""Shared PyMeshLab availability state for the Blender add-on.

pymeshlab ships as a bundled wheel that Blender extracts into its shared
extensions site-packages. Depending on installation order, the wheel may
only become importable *after* the add-on is registered (e.g. right after
installing the extension, or if a previous extraction was rolled back and
is repeated on the next start). The import is therefore attempted lazily
and retried, instead of once at module import time.
"""

import pathlib
import sys
import time


_WINDOWS_PLUGIN_FILES = (
    "filter_meshing.dll",
    "filter_clean.dll",
    "filter_screened_poisson.dll",
    "filter_mesh_alpha_wrap.dll",
)

_module = None
_error = None
_last_attempt = 0.0


def _bind_loaded_filters(pymeshlab_module):
    for filter_name in pymeshlab_module.filter_list():
        setattr(
            pymeshlab_module.MeshSet,
            filter_name,
            pymeshlab_module.bind_function(filter_name),
        )


def _ensure_windows_plugins(pymeshlab_module):
    plugin_dir = pathlib.Path(pymeshlab_module.__file__).resolve().parent / "lib" / "plugins"
    missing = [name for name in _WINDOWS_PLUGIN_FILES if not (plugin_dir / name).exists()]
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(f"Missing bundled PyMeshLab plugins: {missing_names}")

    loaded = set(getattr(pymeshlab_module, "_btc_loaded_plugins", ()))
    for plugin_name in _WINDOWS_PLUGIN_FILES:
        if plugin_name not in loaded:
            pymeshlab_module.load_plugin(str(plugin_dir / plugin_name))
            loaded.add(plugin_name)

    pymeshlab_module._btc_loaded_plugins = tuple(sorted(loaded))
    pymeshlab_module._btc_windows_plugins_ready = True
    _bind_loaded_filters(pymeshlab_module)


def _try_import():
    global _module, _error, _last_attempt
    _last_attempt = time.monotonic()
    try:
        import pymeshlab

        if sys.platform == 'win32':
            _ensure_windows_plugins(pymeshlab)
    except ModuleNotFoundError as e:
        _module = None
        _error = (
            f"{e}. The bundled pymeshlab wheel is not installed — "
            "try restarting Blender, or reinstall the add-on."
        )
    except Exception as e:
        _module = None
        _error = str(e)
    else:
        _module = pymeshlab
        _error = None


def get_pymeshlab():
    """Return the pymeshlab module, importing it on first use.

    Retries the import on every call while unavailable, so a wheel that
    Blender extracts after the add-on registered is picked up without a
    restart. Returns None if pymeshlab is unavailable; see import_error().
    """
    if _module is None:
        _try_import()
    return _module


def import_error(retry_interval=5.0):
    """Return the last import error string, or None if pymeshlab loaded.

    Safe to call from UI draw code: while pymeshlab is unavailable, the
    import is retried at most every ``retry_interval`` seconds.
    """
    if _module is not None:
        return None
    if _error is None or (time.monotonic() - _last_attempt) >= retry_interval:
        _try_import()
    return _error
