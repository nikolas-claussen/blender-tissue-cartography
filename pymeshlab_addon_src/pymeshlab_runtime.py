"""Shared PyMeshLab availability state for the Blender add-on."""

import pathlib
import sys


_WINDOWS_PLUGIN_FILES = (
    "filter_meshing.dll",
    "filter_clean.dll",
    "filter_screened_poisson.dll",
    "filter_mesh_alpha_wrap.dll",
)


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


try:
    import pymeshlab

    if sys.platform == 'win32':
        _ensure_windows_plugins(pymeshlab)
    PYMESHLAB_IMPORT_ERROR = None
except Exception as e:
    pymeshlab = None
    PYMESHLAB_IMPORT_ERROR = str(e)
