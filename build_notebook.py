"""Menyusun berkas parts/*.py menjadi satu notebook .ipynb lalu menjalankannya."""
import glob
import os
import sys
import time

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

BASE = os.path.dirname(os.path.abspath(__file__))
NAMA_NB = 'Proyek_Akhir_Sistem_Rekomendasi.ipynb'


def baca_sumber():
    teks = []
    for p in sorted(glob.glob(os.path.join(BASE, 'parts', 'p*.py'))):
        teks.append(open(p, encoding='utf-8').read().rstrip() + '\n')
    return '\n'.join(teks)


def parse_sel(teks):
    sel, mode, buf = [], None, []

    def tutup():
        if mode is None:
            return
        isi = '\n'.join(buf).strip('\n')
        if not isi.strip():
            return
        if mode == 'markdown':
            baris = []
            for b in isi.split('\n'):
                b = b[1:] if b.startswith('#') else b
                baris.append(b[1:] if b.startswith(' ') else b)
            sel.append(new_markdown_cell('\n'.join(baris).strip('\n')))
        else:
            sel.append(new_code_cell(isi))

    for baris in teks.split('\n'):
        if baris.startswith('# %%'):
            tutup()
            mode = 'markdown' if '[markdown]' in baris else 'code'
            buf = []
        else:
            buf.append(baris)
    tutup()
    return sel


def main():
    nb = new_notebook(cells=parse_sel(baca_sumber()))
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python', 'version': sys.version.split()[0]}

    n_md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
    n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f'Notebook tersusun: {len(nb.cells)} sel ({n_md} markdown, {n_code} kode)')

    jalur = os.path.join(BASE, NAMA_NB)
    nbformat.write(nb, jalur)

    if '--no-exec' in sys.argv:
        return

    from nbclient import NotebookClient
    print('Menjalankan notebook ...')
    t0 = time.time()
    klien = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': BASE}})
    klien.execute()
    nbformat.write(nb, jalur)
    print(f'Selesai dalam {time.time() - t0:.0f} detik -> {jalur}')


if __name__ == '__main__':
    main()
