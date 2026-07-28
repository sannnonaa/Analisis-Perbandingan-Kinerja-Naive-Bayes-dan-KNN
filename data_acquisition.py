import os
import re
import pandas as pd
from ocr_extraction import extract_text, IMAGE_EXTENSIONS

# Konfigurasi deteksi source_type
_SOURCE_KEYWORDS: dict[str, list[str]] = {
    "synthetic": [
        "sintetis", "synthetic", "augmentasi", "augment", "augmented",
        "generated", "buatan", "fake",
    ],
    "real": [
        "riil", "real", "asli", "original", "genuine",
        "pt_atika", "atika",
    ],
}

# Pola regex.
_SOURCE_PATTERNS: dict[str, re.Pattern] = {
    src: re.compile(
        r"(?<![a-z0-9_])(" + "|".join(re.escape(kw) for kw in kws) + r")(?![a-z0-9_])",
        re.IGNORECASE,
    )
    for src, kws in _SOURCE_KEYWORDS.items()
}


def _detect_source_type(path: str) -> str:
    # Normalisasi separator agar konsisten di semua OS
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")

    for part in parts:
        for src, pattern in _SOURCE_PATTERNS.items():
            if pattern.search(part):
                return src
    return "unknown"


def build_dataset_from_folder(base_path, lang="ind"):
    """
    Membaca semua file PDF/DOCX dari folder dataset.
    Nama sub-folder otomatis menjadi label/kategori dokumen.
    """
    rows = []

    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)

        if not os.path.isdir(label_path):
            continue

        for root, dirs, files in os.walk(label_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".pdf", ".docx") | IMAGE_EXTENSIONS:
                    continue

                file_path = os.path.join(root, fname)
                try:
                    text, metode = extract_text(file_path, lang=lang)
                except Exception as exc:
                    # Fallback: error ditangkap di sini sebagai jaring pengaman
                    # (seharusnya sudah ditangani di dalam extract_text)
                    text   = f"GAGAL: {type(exc).__name__}: {exc}"
                    metode = "Digital-Error"
                    print(f"Warning: gagal mengekstrak teks dari {file_path}: {exc}")

                src        = _detect_source_type(file_path)
                is_img     = ext in IMAGE_EXTENSIONS
                is_success = not metode.endswith("-Error") and metode != "Unsupported"

                rows.append({
                    "filename":    fname,
                    "path":        file_path,
                    "label":       label,
                    "text_raw":    text if is_success else "",
                    "metode_ocr":  metode,
                    "source_type": src,
                    "file_type":   "image" if is_img else ext.lstrip("."),
                    "ocr_error":   "" if is_success else text,
                })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    dataset_path = "dataset"
    df = build_dataset_from_folder(dataset_path)
    print(df.head())