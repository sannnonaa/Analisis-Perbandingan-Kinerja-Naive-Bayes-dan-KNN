from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import List

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        StopWordRemoverFactory,
        StopWordRemover,
        ArrayDictionary,
    )
    _SASTRAWI_AVAILABLE = True
except ImportError:
    _SASTRAWI_AVAILABLE = False
    import warnings
    warnings.warn(
        "PySastrawi tidak ter-install. Stemming dan stopword removal dinonaktifkan. "
        "Jalankan: pip install PySastrawi",
        ImportWarning,
        stacklevel=2,
    )

# Kamus normalisasi singkatan → kata lengkap
NORMALIZATION_DICT: dict[str, str] = {
    # Singkatan umum dokumen
    "yth":   "yang terhormat",
    "Yth":   "Yang Terhormat",
    "yg":    "yang",
    "dg":    "dengan",
    "dgn":   "dengan",
    "kpd":   "kepada",
    "utk":   "untuk",
    "tdk":   "tidak",
    "tdk.":  "tidak",
    "krn":   "karena",
    "dll":   "dan lain lain",
    "dsb":   "dan sebagainya",
    "tsb":   "tersebut",
    "svp":   "surat persetujuan",
    "spb":   "surat persetujuan berlayar",
    "ttd":   "tanda tangan",
    "acc":   "disetujui",
    "no":    "nomor",
    "no.":   "nomor",
    "tgl":   "tanggal",
    "tgl.":  "tanggal",
    "hr":    "hari",
    "bln":   "bulan",
    "th":    "tahun",
    "thn":   "tahun",
    "jml":   "jumlah",
    "jlh":   "jumlah",
    "kpl":   "kapal",
    "mbl":   "mobil",
    "tlp":   "telepon",
    "telp":  "telepon",
    "hp":    "handphone",
    "hal":   "halaman",
    "hlm":   "halaman",
    "vol":   "volume",
    "dok":   "dokumen",
    "dok.":  "dokumen",
    "ref":   "referensi",
    "ref.":  "referensi",
    "nib":   "nomor induk berusaha",
    "siup":  "surat izin usaha perdagangan",
    "npwp":  "nomor pokok wajib pajak",
    # Singkatan maritim
    "gt":    "gross tonnage",
    "nt":    "net tonnage",
    "dwt":   "deadweight tonnage",
    "imo":   "international maritime organization",
    "pkk":   "pas kecil kapal",
    "bst":   "basic safety training",
    "ankapin": "ahli nautika kapal penangkap ikan",
    "atkapin": "ahli teknika kapal penangkap ikan",
}

# Stopword custom domain maritim
custom_stopwords: list[str] = [
    # Artefak dokumen
    "ttd", "acc", "yth", "kepada", "perihal", "lampiran", "halaman",
    "nomor", "tanggal", "tahun", "bulan", "hari",
    # Kata generik yang tidak diskriminatif antar kelas
    "surat", "dokumen", "berdasarkan", "bahwa", "dengan", "dalam",
    "untuk", "kepada", "dari", "dan", "atau", "yang", "pada", "ini",
    "itu", "juga", "sudah", "telah", "akan", "ada", "tidak", "serta",
    "oleh", "tersebut", "dapat", "agar", "sebagai", "sesuai",
    # Tambahan kata generik bahasa Indonesia (frekuensi tinggi, nilai diskriminasi rendah)
    "adapun", "akhir", "antar", "antara", "atas", "bagi", "bahwa",
    "baru", "beberapa", "belum", "bila", "bisa", "buat", "cara",
    "demikian", "dengan", "diantara", "dimana", "disamping", "dst",
    "hal", "harus", "hingga", "ialah", "ini", "jadi", "jika",
    "justru", "karena", "kami", "kamu", "kami", "ke", "kita",
    "lagi", "lain", "lalu", "lebih", "makin", "maka", "masih",
    "melalui", "memang", "mengenai", "mengingat", "menjadi",
    "mereka", "misal", "mohon", "mulai", "namun", "nanti",
    "paling", "penting", "pernah", "perlu", "pihak", "pun",
    "sama", "sampai", "saya", "sebelum", "sebuah", "secara",
    "sedang", "setelah", "setiap", "siap", "sudah", "supaya",
    "tentang", "terkait", "terlampir", "termasuk", "terus",
    "tidak", "tiga", "tiap", "turut", "walau", "wajib",
    # Kata administrasi umum (muncul di semua jenis dokumen, DF tinggi)
    "administrasi", "ketentuan", "peraturan", "pelaksanaan",
    "pemenuhan", "penyampaian", "pengurusan", "pengajuan",
    "permohonan", "terlampir", "disampaikan", "dimaksud",
    "ditujukan", "bersama", "bersangkutan", "ditetapkan",
    # Entitas perusahaan yang muncul di semua kelas (tidak diskriminatif)
    "atika", "jaya", "samudera", "pt", "cv",
    # Kata bilangan / satuan
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan",
    "sembilan", "sepuluh", "sebelas", "dua belas",
]

# ---------------------------------------------------------------------------
# Stopword bahasa Inggris — noise dari cookie-consent / privacy-policy yang
# ter-OCR masuk ke dokumen. Disimpan terpisah agar mudah diaudit / dinonaktifkan.
# ---------------------------------------------------------------------------
ENGLISH_NOISE_STOPWORDS: set[str] = {
    # Kata-kata dari tabel IDF yang jelas bukan istilah maritim Indonesia
    "above", "access", "accordance", "actively", "address", "ads",
    "advertising", "agent", "and", "based", "browser", "by",
    "can", "choose", "click", "consent", "content", "cookie",
    "data", "device", "for", "from", "have", "how", "if",
    "including", "information", "interest", "legitimate",
    "list", "may", "more", "object", "of", "on", "or",
    "our", "partners", "personalized", "policy", "privacy",
    "process", "purposes", "scan", "settings", "some",
    "stored", "that", "the", "their", "these", "this",
    "to", "use", "used", "vendor", "we", "website",
    "what", "when", "where", "which", "who", "with", "you", "your",
}

MARITIME_WHITELIST: set[str] = {
    "kapal", "pelabuhan", "nahkoda", "mualim", "masinis", "awak", "pelaut",
    "berlayar", "sandar", "labuh", "dermaga", "tambat", "bongkar", "muat",
    "kargo", "muatan", "manifest", "clearance", "dokcing", "docking",
    "reparasi", "perbaikan", "klasifikasi", "sertifikat", "perizinan",
    "persetujuan", "surveyor", "inspeksi", "audit", "keselamatan",
    "navigasi", "bendera", "registrasi", "gross", "tonnage", "deadweight",
    # Singkatan teknis yang sudah dinormalisasi
    "imo", "bst", "ankapin", "atkapin",
}

_PROPER_NOUN_PREFIXES: set[str] = {
    "mv", "mt", "kmp", "km", "spb", "kt", "ks",   # prefix nama kapal
    "bapak", "ibu", "pak", "bu", "dr", "ir",        # prefix nama orang
    "kapten", "nakhoda",
}

if _SASTRAWI_AVAILABLE:
    _stemmer = StemmerFactory().create_stemmer()

    # Gabungkan stopword Sastrawi default + custom domain maritim
    _sw_factory    = StopWordRemoverFactory()
    _default_sw    = _sw_factory.get_stop_words()
    _combined_sw   = list(set(_default_sw) | set(custom_stopwords))
    _sw_dictionary = ArrayDictionary(_combined_sw)
    _sw_remover    = StopWordRemover(_sw_dictionary)
else:
    _stemmer    = None
    _sw_remover = None


def case_folding(text: str) -> str:
    """
    Tahap 1: Ubah semua karakter ke huruf kecil.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text.lower()


def cleaning(text: str) -> str:
    """
    Tahap 2: Hapus noise dari teks hasil OCR / ekstraksi dokumen.

    Yang dihapus:
    - URL dan email
    - Nomor halaman (pola: "hal. 1", "page 2", "- 3 -")
    - String base64 panjang (artefak CamScanner)
    - Semua angka (digit murni maupun yang menempel huruf seperti "gt300", "no123")
    - Tanda baca dan karakter non-alfabetis
    - Spasi berlebih
    """
    if not isinstance(text, str):
        return ""

    # URL dan email
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Artefak CamScanner / base64 (string panjang tanpa spasi > 40 karakter)
    text = re.sub(r"\S{41,}", " ", text)

    # Nomor halaman berbagai format (hapus sebelum angka di-strip global)
    text = re.sub(r"\b(hal|page|hlm)\.?\s*\d+\b", " ", text)
    text = re.sub(r"-\s*\d+\s*-", " ", text)

    # FIX: Hapus SEMUA digit di mana pun posisinya
    # — mencakup angka standalone ("123"), menempel huruf ("gt300", "2024abc"),
    #   maupun sisa noise OCR berupa karakter angka acak.
    text = re.sub(r"\d+", " ", text)

    # Hapus tanda baca dan karakter non-alfabetis
    # (setelah angka dihapus, [^a-z\s] sudah cukup — tidak perlu 0-9 lagi)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalisasi spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenization(text: str) -> List[str]:
    """
    Tahap 3: Pisahkan teks menjadi list token berdasarkan spasi.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return text.split()


def normalization(tokens: List[str]) -> List[str]:
    """
    Tahap 4: Ganti singkatan dengan kata lengkap menggunakan NORMALIZATION_DICT.

    Catatan: normalisasi bisa menghasilkan frasa multi-kata (misal "dll" →
    "dan lain lain"). Token multi-kata tersebut dipecah kembali agar pipeline
    berikutnya menerima token tunggal.
    """
    expanded: List[str] = []
    for token in tokens:
        replacement = NORMALIZATION_DICT.get(token, token)
        # Pecah hasil normalisasi jika mengandung spasi
        expanded.extend(replacement.split())
    return expanded


def stopword_removal(tokens: List[str]) -> List[str]:
    """
    Tahap 5: Hapus stopword dan token yang tidak informatif.

    Perbaikan:
    - Filter token yang seluruhnya digit (jaga-jaga jika angka lolos dari cleaning)
    - Filter token yang mengandung digit (campuran huruf-angka sisa OCR)
    - Filter token terlalu pendek (< 3 karakter)
    - Filter bahasa Inggris (noise dari cookie-consent / privacy-policy yang ter-OCR)
    """
    # FIX: buang token yang masih mengandung digit (defense-in-depth)
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

    # Filter token terlalu pendek
    tokens = [t for t in tokens if len(t) >= 3]

    # Buang noise bahasa Inggris (cookie-consent, GDPR boilerplate, dll.)
    # Cek sebelum Sastrawi agar tidak memengaruhi stemmer
    tokens = [t for t in tokens if t not in ENGLISH_NOISE_STOPWORDS]

    if not tokens:
        return []

    if _sw_remover is not None:
        text_joined   = " ".join(tokens)
        text_filtered = _sw_remover.remove(text_joined)
        # Terapkan juga custom_stopwords secara eksplisit setelah Sastrawi
        # (Sastrawi kadang meloloskan kata yang sudah di-stem tapi masih generik)
        sw_set = set(custom_stopwords)
        return [t for t in text_filtered.split() if t not in sw_set]
    else:
        # Fallback tanpa Sastrawi: filter manual dari custom_stopwords
        sw_set = set(custom_stopwords)
        return [t for t in tokens if t not in sw_set]


def proper_noun_removal(tokens: List[str], raw_text: str = "") -> List[str]:
    """
    Tahap 6: Hapus kemungkinan proper noun (nama kapal, nama orang).

    Heuristik yang digunakan:
    A. Token yang mengikuti prefix kapal/orang dalam teks asli (raw_text)
       kemungkinan besar nama → dihapus.
    B. Token yang diawali huruf kapital di teks asli DAN tidak ada di
       MARITIME_WHITELIST → dihapus.
    """
    if not tokens:
        return []

    # Bangun set kata yang dikapitalisasi di raw_text (heuristik B)
    capitalized_in_raw: set[str] = set()
    if raw_text:
        raw_tokens = raw_text.split()
        for i, rt in enumerate(raw_tokens):
            # Token kapital di tengah kalimat (bukan awal kalimat)
            if i > 0 and rt and rt[0].isupper() and len(rt) > 2:
                capitalized_in_raw.add(rt.lower())

    # Bangun set token setelah prefix (heuristik A)
    after_prefix: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok in _PROPER_NOUN_PREFIXES and i + 1 < len(tokens):
            after_prefix.add(tokens[i + 1])

    result = []
    for tok in tokens:
        # Pertahankan kalau ada di whitelist maritim
        if tok in MARITIME_WHITELIST:
            result.append(tok)
            continue
        # Buang kalau teridentifikasi proper noun
        if tok in after_prefix:
            continue
        if tok in capitalized_in_raw:
            continue
        result.append(tok)

    return result


def stemming(tokens: List[str]) -> List[str]:
    """
    Tahap 7: Reduksi kata ke bentuk dasar menggunakan Sastrawi.
    """
    if not tokens:
        return []

    if _stemmer is not None:
        return [_stemmer.stem(tok) for tok in tokens]
    else:
        return tokens


def idf_filter(tokens: List[str], idf_vocab: dict[str, float], min_idf: float = 2.5) -> List[str]:
    """
    Tahap opsional: Buang token dengan IDF di bawah threshold.

    Digunakan setelah pipeline dilatih dan idf_vocab tersedia (misal dari
    TfidfVectorizer.idf_ yang dipetakan ke vocabulary_).

    Parameter
    ---------
    tokens   : list token hasil stemming
    idf_vocab: dict  {kata_stem: nilai_idf}  — diisi dari model TF-IDF terlatih
    min_idf  : threshold minimum IDF; token dengan IDF < min_idf dibuang.
               Default 2.5 membuang kata yang muncul di > ~8% dokumen latih
               (asumsi N=67 dok: IDF < 2.5 ≈ DF > 5).

    Catatan: jika idf_vocab kosong, fungsi mengembalikan tokens apa adanya.
    """
    if not idf_vocab:
        return tokens
    return [t for t in tokens if idf_vocab.get(t, min_idf + 1) >= min_idf]


def preprocess_text(
    text: str,
    raw_text: str = "",
    idf_vocab: dict[str, float] | None = None,
    min_idf: float = 2.5,
) -> str:
    """
    Jalankan seluruh pipeline preprocessing dan kembalikan string bersih.

    Pipeline: case_folding → cleaning → tokenization → normalization
              → stopword_removal → proper_noun_removal → stemming
              → [idf_filter jika idf_vocab disuplai]

    Parameter tambahan
    ------------------
    idf_vocab : dict {kata_stem: idf_value} dari model TF-IDF terlatih.
                Jika disuplai, token dengan IDF < min_idf akan dibuang.
    min_idf   : threshold IDF minimum (default 2.5).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Simpan raw untuk proper_noun_removal jika tidak disuplai eksplisit
    if not raw_text:
        raw_text = text

    cf   = case_folding(text)
    cl   = cleaning(cf)
    tok  = tokenization(cl)
    norm = normalization(tok)
    sw   = stopword_removal(norm)
    pn   = proper_noun_removal(sw, raw_text=raw_text)
    stem = stemming(pn)

    if idf_vocab:
        stem = idf_filter(stem, idf_vocab, min_idf=min_idf)

    return " ".join(stem)


def preprocess_batch(
    texts: list[str],
    raw_texts: list[str] | None = None,
    idf_vocab: dict[str, float] | None = None,
    min_idf: float = 2.5,
) -> list[str]:
    """
    Jalankan preprocess_text untuk sekumpulan teks sekaligus.

    Parameter tambahan
    ------------------
    idf_vocab : dict {kata_stem: idf_value} — opsional, diteruskan ke preprocess_text.
    min_idf   : threshold IDF minimum (default 2.5).
    """
    if raw_texts is None:
        raw_texts = [""] * len(texts)

    if len(texts) != len(raw_texts):
        raise ValueError(
            f"Panjang texts ({len(texts)}) dan raw_texts ({len(raw_texts)}) harus sama."
        )

    return [
        preprocess_text(t, r, idf_vocab=idf_vocab, min_idf=min_idf)
        for t, r in zip(texts, raw_texts)
    ]