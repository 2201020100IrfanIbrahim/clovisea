"""
app.py — Clovisea: deteksi jenis lamun dengan CenterNet.
"""
import os
import io
import time
import base64

from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image

import model as M

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "best.pth")
THRESH_PATH = os.path.join(BASE_DIR, "conf_thresholds.json")


def _to_data_uri(pil_img, quality=88):
    """Encode PIL.Image jadi data URI base64 — tanpa menyimpan file ke disk."""
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_MB = 8

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
app.secret_key = "clovisea-secret-key-ganti-ini"

print("Memuat model CenterNet...")
DEVICE = "cpu"
MODEL = M.load_model(CKPT_PATH, device=DEVICE) if os.path.isfile(CKPT_PATH) else None
THRESH = M.load_thresholds(THRESH_PATH)
print("Model siap." if MODEL else "PERINGATAN: best.pth tidak ditemukan di folder root.")

# Lineage WoRMS bersama untuk semua lamun (diverifikasi via marinespecies.org)
_COMMON = [
    ("Kingdom", "Plantae"), ("Subkingdom", "Viridiplantae"),
    ("Infrakingdom", "Streptophyta"), ("Phylum (Divisi)", "Tracheophyta"),
    ("Subphylum", "Spermatophytina"), ("Class", "Magnoliopsida"),
    ("Superorder", "Lilianae"), ("Order", "Alismatales"),
]

def _tax(family, genus, species):
    return _COMMON + [("Family", family), ("Genus", genus), ("Species", species)]

# id, nama, authority (WoRMS), taksonomi, deskripsi, ciri
SPECIES = [
    {"id": 0, "name": "Cymodocea rotundata",
     "authority": "Ascherson & Schweinfurth, 1870",
     "tax": _tax("Cymodoceaceae", "Cymodocea", "Cymodocea rotundata"),
     "daun": "Berbentuk pita, ujung daun membulat, tepi halus tanpa gerigi.",
     "rhizoma": "Merayap horizontal.", "akar": "Serabut halus.",
     "desc": "Lamun berukuran sedang yang umum ditemukan pada rataan pasir dangkal "
             "di daerah pasang surut. Termasuk spesies yang adaptif terhadap kondisi marginal."},
    {"id": 1, "name": "Enhalus acoroides",
     "authority": "(Linnaeus f.) Royle, 1839",
     "tax": _tax("Hydrocharitaceae", "Enhalus", "Enhalus acoroides"),
     "daun": "Sangat panjang menyerupai pita, lebar, dapat melebihi 1 meter.",
     "rhizoma": "Tebal dengan serat kasar.", "akar": "Tebal dan panjang.",
     "desc": "Lamun terbesar di antara jenis lamun tropis dan sering mendominasi padang "
             "lamun. Tumbuh di substrat lunak berlumpur maupun berpasir."},
    {"id": 2, "name": "Halodule uninervis",
     "authority": "(Forsskål) Ascherson, 1882",
     "tax": _tax("Cymodoceaceae", "Halodule", "Halodule uninervis"),
     "daun": "Tipis dan sempit dengan ujung bergerigi tiga (tridentate).",
     "rhizoma": "Halus, merayap.", "akar": "Serabut tipis.",
     "desc": "Daunnya yang tipis dan kecil membuatnya menjadi salah satu jenis lamun "
             "yang paling sulit dideteksi pada citra bawah air."},
    {"id": 3, "name": "Halophila ovalis",
     "authority": "(R.Brown) J.D.Hooker, 1858",
     "tax": _tax("Hydrocharitaceae", "Halophila", "Halophila ovalis"),
     "daun": "Oval kecil menyerupai sendok, memiliki tulang daun melintang yang khas.",
     "rhizoma": "Merayap dengan ruas pendek.", "akar": "Tunggal pada tiap ruas.",
     "desc": "Bentuk daun oval yang khas membuat jenis ini relatif mudah dikenali oleh model "
             "dibanding jenis lamun lain."},
    {"id": 4, "name": "Syringodium isoetifolium",
     "authority": "(Ascherson) Dandy, 1939",
     "tax": _tax("Cymodoceaceae", "Syringodium", "Syringodium isoetifolium"),
     "daun": "Silindris menyerupai jarum atau lidi (bukan pipih).",
     "rhizoma": "Merayap.", "akar": "Serabut.",
     "desc": "Tumbuh rapat dengan daun berbentuk silindris yang menjadi ciri pembeda "
             "utama dari jenis lamun berdaun pipih."},
    {"id": 5, "name": "Thalassia hemprichii",
     "authority": "(Ehrenberg ex Solms) Ascherson, 1871",
     "tax": _tax("Hydrocharitaceae", "Thalassia", "Thalassia hemprichii"),
     "daun": "Melengkung menyerupai sabit, terdapat bintik-bintik (sel tanin).",
     "rhizoma": "Tebal dengan bekas pelepah daun.", "akar": "Serabut tebal.",
     "desc": "Salah satu jenis lamun paling umum di perairan tropis dan sering ditemukan "
             "bercampur dengan jenis lamun lainnya."},
]
SPECIES_BY_ID = {s["id"]: s for s in SPECIES}

AUTHOR = {
    "nama": "Irfan Ibrahim",
    "nim": "2201020100",
    "jurusan": "Teknik Informatika",
    "kampus": "Universitas Maritim Raja Ali Haji",
    "bio": "Mahasiswa Teknik Informatika yang mengembangkan Clovisea sebagai bagian "
           "dari penelitian skripsi mengenai identifikasi jenis lamun di perairan "
           "Pulau Bintan menggunakan metode deep learning (CenterNet).",
}

LOCATION = {
    "nama": "Desa Pengudang, Kabupaten Bintan, Kepulauan Riau",
    "desc": "Penelitian dilakukan di kawasan padang lamun perairan Desa Pengudang, Pulau Bintan, "
    "Kepulauan Riau. Wilayah ini memiliki ekosistem padang lamun yang beragam dan menjadi habitat "
    "penting bagi berbagai jenis lamun tropis. Pengambilan data citra dilakukan langsung di lapangan "
    "(in situ) di bawah permukaan air. Desa Pengudang dipilih karena padang lamun di wilayah ini rentan "
    "mengalami kerusakan, terutama akibat tumpahan limbah minyak. Merespons kondisi tersebut, pemerintah "
    "telah menetapkannya sebagai bagian dari kawasan konservasi yang wajib dipantau secara rutin. Sayangnya, "
    "pemantauan manual oleh penyelam saat ini masih memakan waktu, tenaga, dan rentan terjadi kesalahan pencatatan. "
    "Oleh karena itu, penelitian di lokasi ini bertujuan untuk menerapkan teknologi yang bisa mengenali jenis lamun "
    "secara otomatis, sehingga proses pengawasan area konservasi ke depannya bisa jauh lebih cepat dan akurat.",
}


def _allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html", species=SPECIES)


@app.route("/detect", methods=["POST"])
def detect():
    if MODEL is None:
        flash("Model belum tersedia di server (best.pth tidak ditemukan).")
        return redirect(url_for("index"))

    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Silakan pilih atau ambil foto terlebih dahulu.")
        return redirect(url_for("index"))
    if not _allowed(file.filename):
        flash("Format tidak didukung. Gunakan .jpg, .jpeg, atau .png.")
        return redirect(url_for("index"))
    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        flash("Gagal membaca gambar. Pastikan file valid.")
        return redirect(url_for("index"))

    t0 = time.time()
    dets, hm = M.predict(MODEL, pil_img, THRESH, device=DEVICE)
    bbox_img = M.draw_detections(pil_img, dets)
    heat_img = M.draw_heatmap_overlay(pil_img, hm)
    elapsed = time.time() - t0

    # tidak menyimpan ke disk — kirim langsung sebagai data URI base64
    bbox_uri = _to_data_uri(bbox_img)
    heat_uri = _to_data_uri(heat_img)

    # ringkasan per kelas + taksonomi untuk dropdown
    counts = {}
    for d in dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    summary = []
    for cls_id, n in sorted(counts.items(), key=lambda x: -x[1]):
        sp = SPECIES_BY_ID[cls_id]
        summary.append({"name": sp["name"], "count": n, "color": M.CLASS_COLORS[cls_id],
                        "authority": sp["authority"], "tax": sp["tax"]})

    return render_template("result.html",
                           bbox_image=bbox_uri,
                           heat_image=heat_uri,
                           summary=summary, count=len(dets), elapsed=f"{elapsed:.2f}")


@app.route("/species")
def species_list():
    return render_template("species.html", species=SPECIES)


@app.route("/species/<int:sid>")
def species_detail(sid):
    sp = SPECIES_BY_ID.get(sid)
    if not sp:
        return redirect(url_for("species_list"))
    return render_template("species_detail.html", s=sp)


@app.route("/location")
def location():
    return render_template("location.html", loc=LOCATION)


@app.route("/author")
def author():
    return render_template("author.html", a=AUTHOR)


@app.errorhandler(413)
def too_large(e):
    flash(f"Ukuran file terlalu besar (maksimum {MAX_MB} MB).")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
