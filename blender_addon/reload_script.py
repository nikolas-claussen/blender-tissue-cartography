import sys, importlib, bpy

pkg = "bl_ext.user_default.tissue_cartography"
for key in sorted(
    (k for k in sys.modules if k == pkg or k.startswith(pkg + ".")),
    key=lambda k: -k.count(".")
):
    importlib.reload(sys.modules[key])

bpy.ops.preferences.addon_disable(module=pkg)
bpy.ops.preferences.addon_enable(module=pkg)