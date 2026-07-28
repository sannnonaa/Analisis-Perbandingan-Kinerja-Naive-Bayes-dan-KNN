#ocr_extraction.py

import os
import fitz  # PyMuPDF
import pytesseract
from docx import Document
import cv2
import numpy as np
import tempfile
import shutil

_WINDOWS_FALLBACKS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\capspro\tesseract-ocr\tesseract.exe",   # lokasi lama — tetap sebagai last resort
]

def _find_tesseract_exe() -> str:
    """Cari executable Tesseract secara portabel."""
    # 1. Environment variable eksplisit
    from_env = os.getenv("TESSERACT_EXE", "").strip()
    if from_env:
        return from_env
    # 2. PATH sistem (Linux, macOS, Windows yang sudah ditambah ke PATH)
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    # 3. Fallback lokasi Windows umum
    for path in _WINDOWS_FALLBACKS:
        if os.path.isfile(path):
            return path
    return ""   # tidak ditemukan

def _find_tessdata_dir(exe_path: str) -> str:
    """Cari folder tessdata: env var, lalu satu level di atas exe."""
    from_env = os.getenv("TESSDATA_DIR", "").strip()
    if from_env and os.path.isdir(from_env):
        return from_env
    # Tebak dari lokasi exe: biasanya <install_dir>/tessdata
    if exe_path:
        candidate = os.path.join(os.path.dirname(exe_path), "tessdata")
        if os.path.isdir(candidate):
            return candidate
    return ""

TESSERACT_EXE  = _find_tesseract_exe()
TESSDATA_DIR   = _find_tessdata_dir(TESSERACT_EXE)

if TESSERACT_EXE and os.path.isfile(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
else:
    print(
        "[OCR] Peringatan: Tesseract tidak ditemukan secara otomatis.\n"
        "  Solusi:\n"
        "    1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "    2. Set environment variable:  TESSERACT_EXE=<path ke tesseract.exe>\n"
        "  OCR dokumen scan akan gagal sampai Tesseract dikonfigurasi."
    )

if TESSDATA_DIR and os.path.isdir(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
else:
    print(
        "[OCR] Peringatan: folder tessdata tidak ditemukan.\n"
        "  Set environment variable:  TESSDATA_DIR=<path ke folder tessdata>\n"
        "  Dokumen berbahasa Indonesia mungkin tidak ter-OCR dengan benar."
    )

def is_scanned_text(text, min_len=150, min_alpha_ratio=0.40):
    """
    Mendeteksi apakah PDF adalah dokumen scan (image-based) atau digital.

    Dua kondisi dianggap scan dan perlu OCR:
    1. Teks terlalu pendek (< min_len karakter) — PDF murni gambar tanpa text layer.
    2. Rasio karakter alfabetis terlalu rendah (< min_alpha_ratio) — PDF dengan
       text layer otomatis dari CamScanner/Adobe Scan yang menghasilkan noise
       karakter acak, bukan teks bermakna.
    """
    if text is None:
        return True
    stripped = text.strip()
    # Kondisi 1: teks terlalu pendek
    if len(stripped) < min_len:
        return True
    # Kondisi 2: rasio alfabetis terlalu rendah (noise dari text layer CamScanner)
    non_space = [c for c in stripped if not c.isspace()]
    if not non_space:
        return True
    alpha_ratio = sum(1 for c in non_space if c.isalpha()) / len(non_space)
    if alpha_ratio < min_alpha_ratio:
        return True
    return False

def preprocess_image_for_ocr(img_array):
    if len(img_array.shape) == 3:
        # Handle RGBA (4 channel) dari PyMuPDF sebelum konversi ke grayscale
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Thresholding (Otsu) agar teks hitam putih tegas
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def extract_text_from_pdf_hybrid(file_path, lang="ind"):
    """
    Ekstraksi teks PDF menggunakan metode Hybrid (Digital + OCR via Memory).
    """
    text   = ""
    metode = "Digital"  # asumsi awal; diubah jika fallback ke OCR
    try:
        with fitz.open(file_path) as doc:
            # Tahap 1: ambil teks digital
            for page in doc:
                text += page.get_text()

            # Tahap 2: jika teks kosong, sedikit, atau rasio alfabet rendah
            # (noise dari text layer CamScanner) → jalankan OCR
            if is_scanned_text(text):
                metode    = "OCR"   # catat SEBELUM teks ditimpa hasil OCR
                ocr_texts = []
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI

                    # ubah sampel jadi numpy array
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.h, pix.w, pix.n)

                    # preprocessing
                    img_pre = preprocess_image_for_ocr(img)

                    # tesseract
                    t = pytesseract.image_to_string(img_pre, lang=lang)
                    ocr_texts.append(t)

                text = "\n".join(ocr_texts)
    except Exception as e:
        print(f"Error pada {os.path.basename(file_path)}: {e}")
        text   = ""
        metode = "Digital"  # gagal total — tetap anggap non-OCR

    return text, metode

# Format gambar yang didukung
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
    ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".jp2",
}


def extract_text_from_image(file_path, lang="ind"):
    """
    Ekstraksi teks dari file gambar (PNG, JPG, BMP, TIFF, WEBP, GIF, dll.)
    menggunakan Tesseract OCR.

    Mengembalikan (text, metode) atau (pesan_error, "OCR-Error") jika gagal.
    """
    # Pastikan Tesseract sudah dikonfigurasi
    if not TESSERACT_EXE:
        return (
            "GAGAL: Tesseract tidak ditemukan. "
            "Install Tesseract dan set TESSERACT_EXE.",
            "OCR-Error",
        )

    try:
        from PIL import Image as PILImage
    except ImportError:
        return "GAGAL: Pillow tidak terinstall (pip install Pillow).", "OCR-Error"

    try:
        pil_img = PILImage.open(file_path)

        # GIF: ambil frame pertama saja
        if getattr(pil_img, "is_animated", False):
            pil_img.seek(0)

        # Konversi ke RGB agar OpenCV/numpy tidak protes (RGBA, P, L, dll.)
        if pil_img.mode not in ("RGB", "L"):
            pil_img = pil_img.convert("RGB")

        img_array = np.array(pil_img)
        img_pre   = preprocess_image_for_ocr(img_array)
        text      = pytesseract.image_to_string(img_pre, lang=lang)
        return text, "OCR"

    except pytesseract.TesseractNotFoundError:
        return (
            "GAGAL: Tesseract tidak ditemukan saat runtime. "
            "Pastikan Tesseract terinstall dan ada di PATH.",
            "OCR-Error",
        )
    except FileNotFoundError:
        return f"GAGAL: File tidak ditemukan — {file_path}", "OCR-Error"
    except Exception as exc:
        return f"GAGAL ekstraksi gambar: {type(exc).__name__}: {exc}", "OCR-Error"


def extract_text(file_path, lang="ind"):
    """
    Ekstraksi teks dari PDF, DOCX, atau file gambar
    (PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF, dll.).

    Mengembalikan tuple (text: str, metode: str).
    Jika terjadi error, mengembalikan pesan error sebagai teks
    dengan metode "Digital-Error" atau "OCR-Error" agar pipeline
    tetap berjalan tanpa crash.
    """
    if not file_path or not os.path.exists(file_path):
        return f"GAGAL: File tidak ditemukan — {file_path}", "Digital-Error"

    temp_dir       = tempfile.gettempdir()
    # Hindari collision nama file jika ada file berbeda dengan nama sama
    unique_name    = f"{os.getpid()}_{os.path.basename(file_path)}"
    temp_file_path = os.path.join(temp_dir, unique_name)

    try:
        shutil.copy2(file_path, temp_file_path)
    except Exception as exc:
        return f"GAGAL menyalin file sementara: {exc}", "Digital-Error"

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            text, metode = extract_text_from_pdf_hybrid(temp_file_path, lang)

        elif ext == ".docx":
            try:
                doc    = Document(temp_file_path)
                text   = "\n".join([p.text for p in doc.paragraphs])
                metode = "Digital"
            except Exception as exc:
                text   = f"GAGAL membaca DOCX: {exc}"
                metode = "Digital-Error"

        elif ext in IMAGE_EXTENSIONS:
            text, metode = extract_text_from_image(temp_file_path, lang)

        else:
            text   = (
                f"Format tidak didukung: '{ext}'. "
                f"Format yang didukung: PDF, DOCX, "
                + ", ".join(sorted(IMAGE_EXTENSIONS)).upper().replace(".", "")
            )
            metode = "Unsupported"

    except Exception as exc:
        text   = f"GAGAL TOTAL: {type(exc).__name__}: {exc}"
        metode = "Digital-Error"

    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except OSError:
            pass  # File sementara tidak kritis jika gagal dihapus

    return text, metode