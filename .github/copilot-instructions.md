#  Copilot Instructions (Repository-wide)

## Project context

This project comprises an 3d-image processing software library for "tissue cartography", a method that extracts 2D surfaces from 3D image data and maps them to 2D. It has two components: a standalone python library, and an add-on for the 3D modelling software Blender.

## Coding standards (applies to all Python)

- Use **Python 3.10+**.
- Prefe **concise** code; avoid boiler-plate, and unccessary helper variables and functions.
- Use **docstrings** (NumPy style). Include units and parameter domains when relevant.
- Follow standard **PEP8** style guidelines. Lint with `ruff`.


## Virtual environment, packaging & docs (python library)

- This repo contains large files (image test data, meshes). These are tracked using git LFS (see `.gitattributes`).
- The python library is developped in **Jupyter Notebooks** using **nbdev**. These notebooks live in the `nbdev` folder and are also used to generate the documentation webpage.
- Use nbdev to export code via the `ndbdev_export` command. Do not edit the code files in `blender_tissue_cartography/` directly. Cells to be exported should be marked with #| export at the top. To generate documentation, use `nbdev_docs` and `nbdev_readme`. Nbdev places docs in the `_docs` folder. To update the documentation webpage, delete the old `docs` folder, run the nbdev commands, then move `_docs` to `docs`.
- In notebooks, use separate cells for defining functions/classes and for running code. Below a cell that defines a function/class, include a test cell that runs basic tests or examples of usage.

## Blender add-on

- The blender add-on is in the `blender_addon` folder. It is written in Python, but follows Blender's coding style and conventions, which are different from the rest of the project. These `.py` files are not generated from notebooks, and can be edited directly.

- The add-on must be packaged into an installable Blender add-on,
The `blender_addon/wheels/` directory contains the python libraries the add-on depends on. To build the package, run the shell command:
```sh
/Applications/Blender.app/Contents/MacOS/Blender --command extension build --source-dir . --output-dir addon_zips/ --split-platforms
```
To re-download the wheels, run this command [to do]:
```sh
```