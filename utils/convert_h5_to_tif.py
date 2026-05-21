import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Listbox, Button, Scrollbar
import h5py
import tifffile
import numpy as np

def select_file():
    path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
    if not path:
        return

    # Step 1: Get dataset names
    with h5py.File(path, 'r') as f:
        datasets = []
        def collect_dsets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        f.visititems(collect_dsets)

    if not datasets:
        messagebox.showerror("Error", "No datasets found.")
        return

    # Step 2: Let user choose one
    def on_select():
        sel = listbox.curselection()
        if not sel:
            return
        ds_name = datasets[sel[0]]
        popup.destroy()

        # Step 3: Open file again to read dataset
        with h5py.File(path, 'r') as f:
            data = f[ds_name][()]

        save_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF", "*.tif")])
        if not save_path:
            return

        # Optional: guess axes for metadata
        axes = 'TZCYX'[-data.ndim:]
        tifffile.imwrite(save_path, data, #metadata={'axes': axes},
        	imagej=True)
        messagebox.showinfo("Success", f"Saved to {save_path}")

    popup = Toplevel(root)
    popup.title("Select Dataset")
    listbox = Listbox(popup, width=60, height=20)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = Scrollbar(popup, command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)
    for ds in datasets:
        listbox.insert("end", ds)
    Button(popup, text="Select", command=on_select).pack(pady=5)

root = tk.Tk()
root.title("HDF5 to TIFF Converter")
Button(root, text="Open .h5 and Export Dataset", command=select_file).pack(padx=20, pady=20)
root.mainloop()

