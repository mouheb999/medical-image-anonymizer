import os, glob

rapport_dir = r"c:\Users\MSI\Desktop\PFE_Test\rapport"
tex_files = [rapport_dir + r"\main.tex"] + glob.glob(rapport_dir + r"\chapters\*.tex")

BOM = b'\xef\xbb\xbf'

for path in tex_files:
    data = open(path, 'rb').read()
    if data.startswith(BOM):
        print(f"BOM FOUND and stripped: {path}")
        open(path, 'wb').write(data[3:])
    else:
        print(f"Clean (no BOM): {os.path.basename(path)} -- first bytes: {list(data[:4])}")
