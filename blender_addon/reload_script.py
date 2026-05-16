import bpy, importlib, sys

addon_pkg = "tissue_cartography"

# Reload all submodules (leaf modules first, then the package)
for key in sorted(
    (k for k in sys.modules if k == addon_pkg or k.startswith(addon_pkg + ".")),
    key=lambda k: -k.count(".")   # submodules before package
):
    importlib.reload(sys.modules[key])

# Re-run unregister/register
bpy.ops.preferences.addon_disable(module=addon_pkg)
bpy.ops.preferences.addon_enable(module=addon_pkg)