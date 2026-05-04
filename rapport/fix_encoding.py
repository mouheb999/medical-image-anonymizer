import os, glob

rapport_dir = r"c:\Users\MSI\Desktop\PFE_Test\rapport"
tex_files = [rapport_dir + r"\main.tex"] + glob.glob(rapport_dir + r"\chapters\*.tex")

for path in tex_files:
    raw = open(path, 'rb').read()
    size = len(raw)
    bom = raw[:3] == b'\xef\xbb\xbf'
    has_crlf = b'\r\n' in raw
    has_null = b'\x00' in raw
    content = raw[3:] if bom else raw
    # Strip BOM and normalize to LF
    content = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    # Remove null bytes
    content = content.replace(b'\x00', b'')
    
    changed = (bom or has_crlf or has_null or len(content) != len(raw))
    if changed:
        open(path, 'wb').write(content)
    
    name = os.path.basename(path)
    print(f"{name}: size={size}, BOM={bom}, CRLF={has_crlf}, null={has_null}, fixed={changed}")

print("\nDone. All files checked.")
