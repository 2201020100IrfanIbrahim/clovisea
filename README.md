# SiLamun — Deteksi Jenis Lamun (CenterNet)

Aplikasi web Flask untuk mendeteksi 6 jenis lamun menggunakan model CenterNet (ResNet-50).

## Menjalankan lokal
```
pip install -r requirements.txt
python app.py
```
Buka http://localhost:5000

## Struktur
- app.py            : server Flask (routes)
- model.py          : arsitektur CenterNet + decode + inferensi (PIL)
- best.pth          : bobot model (LETAKKAN MANUAL di sini, ~125MB)
- conf_thresholds.json : ambang confidence per-kelas
- templates/        : halaman HTML
- static/           : CSS + folder upload hasil

## Deploy ke Render.com
1. Push folder ini ke GitHub (TERMASUK best.pth — lihat catatan Git LFS di bawah).
2. Render -> New -> Web Service -> hubungkan repo.
3. Render membaca render.yaml otomatis. Plan: Free.

CATATAN best.pth (~125MB): GitHub menolak file > 100MB via push biasa.
Gunakan Git LFS:
```
git lfs install
git lfs track "*.pth"
git add .gitattributes best.pth
git commit -m "add model"
git push
```
