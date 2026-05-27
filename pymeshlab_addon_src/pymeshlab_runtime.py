"""Shared PyMeshLab availability state for the Blender add-on."""

import sys


if sys.platform == 'win32':
    PYMESHLAB_IMPORT_ERROR = (
        "PyMeshLab is not supported on Windows: its Qt5 filter plugins cause a "
        "fatal DLL crash in Blender's process (MSVCP140 TLS conflict). "
        "See pymeshlab issue #398."
    )
    pymeshlab = None
else:
    try:
        import pymeshlab
        PYMESHLAB_IMPORT_ERROR = None
    except Exception as e:
        pymeshlab = None
        PYMESHLAB_IMPORT_ERROR = str(e)
