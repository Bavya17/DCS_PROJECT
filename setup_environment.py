import os
import sys
import shutil
import site

# Locate active site-packages directory
site_packages = None
for p in site.getsitepackages():
    if "site-packages" in p:
        site_packages = p
        break

if not site_packages or not os.path.exists(site_packages):
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")

base_dir = os.path.join(site_packages, "compressai")
src_shim = os.path.join(os.path.dirname(__file__), "src", "compressai_shim")

print(f"Installing CompressAI pure-PyTorch shim into: {base_dir}")
os.makedirs(os.path.join(base_dir, "layers"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "ops"), exist_ok=True)

with open(os.path.join(base_dir, "__init__.py"), "w", encoding="utf-8") as f:
    f.write('__version__ = "1.2.8"\n')

# Copy all verified pure-PyTorch implementation files from src/compressai_shim
for folder in ["ops", "layers"]:
    src_folder = os.path.join(src_shim, folder)
    dst_folder = os.path.join(base_dir, folder)
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if fname.endswith(".py"):
            shutil.copy2(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

print("CompressAI pure-PyTorch layer package installed successfully!")
