# for internal, testing use: run this script in Blender's scripting workspace
# to reload the addon after making changes to the code.

import sys, importlib, bpy

pkg = "bl_ext.user_default.pymeshlab_remesh_interactive"
for key in sorted(
    (k for k in sys.modules if k == pkg or k.startswith(pkg + ".")),
    key=lambda k: -k.count(".")
):
    importlib.reload(sys.modules[key])

bpy.ops.preferences.addon_disable(module=pkg)
bpy.ops.preferences.addon_enable(module=pkg)