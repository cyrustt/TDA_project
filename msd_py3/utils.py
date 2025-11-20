# utils.py
import os

def get_all_files(basedir, ext='.h5'):
    out = []
    for root, _, files in os.walk(basedir):
        for f in files:
            if f.lower().endswith(ext):
                out.append(os.path.join(root, f))
    out.sort()
    return out
