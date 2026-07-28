import tkinter as tk
from tkinter import font as tkfont
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

#  PALET WARNA
CLR = {
    "bg":           "#F0F2F8", "surface":      "#FFFFFF", "surface2":     "#F8FAFF", "primary":      "#4F46E5", "primary_dk":   "#3730A3",
    "primary_lt":   "#EEF2FF", "success":      "#10B981", "success_lt":   "#D1FAE5", "warning":      "#F59E0B", "warning_lt":   "#FEF3C7",
    "danger":       "#EF4444", "danger_lt":    "#FEE2E2", "purple":       "#8B5CF6", "purple_lt":    "#EDE9FE", "blue":         "#3B82F6",
    "blue_lt":      "#DBEAFE", "teal":         "#14B8A6", "orange":       "#F97316", "text_hd":      "#1E1B4B", "text_body":    "#374151",
    "text_muted":   "#6B7280", "border":       "#E5E7EB", "border_focus": "#A5B4FC",
}

# Data Penelitian
DATASET_INFO = {
    "total":    80,
    "riil":     15,
    "sintetis": 65,
    "kelas": {
        "Sertifikat":  {"total": 20, "riil": 9,  "sintetis": 11},
        "Persetujuan": {"total": 20, "riil": 3,  "sintetis": 17},
        "Perbaikan":   {"total": 20, "riil": 2,  "sintetis": 18},
        "Perizinan":   {"total": 20, "riil": 1,  "sintetis": 19},
    },
}

KELAS_COLORS = {
    "Sertifikat":  CLR["primary"], "Persetujuan": CLR["success"], "Perbaikan":   CLR["warning"], "Perizinan":   CLR["purple"],
}


class BusinessFrame(tk.Frame):
    def __init__(self, parent, controller=None):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller

        self.title_font   = tkfont.Font(family="Segoe UI", size=9,  weight="bold")
        self.card_hd_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.body_font    = tkfont.Font(family="Segoe UI", size=10)
        self.bold_body    = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        self._build_header()
        self._build_scroll_area()
        self._populate()

    #  HEADER FIXED
    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(side="top", fill="x")

        accent = tk.Frame(outer, bg=CLR["primary"], width=5)
        accent.pack(side="left", fill="y")
        accent.pack_propagate(False)

        inner = tk.Frame(outer, bg=CLR["surface"])
        inner.pack(side="left", fill="both", expand=True, padx=20, pady=14)

        tk.Label(inner, text="Business Understanding", font=("Segoe UI", 18, "bold"), bg=CLR["surface"], fg=CLR["text_hd"]).pack(anchor="w")

    #  SCROLL AREA
    def _build_scroll_area(self):
        wrap = tk.Frame(self, bg=CLR["bg"])
        wrap.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(wrap, bg=CLR["bg"], highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=sb.set)

        self.main_container = tk.Frame(self.canvas, bg=CLR["bg"])
        self.canvas_window  = self.canvas.create_window((0, 0), window=self.main_container, anchor="n")

        self.main_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Bug #8 fix: gunakan Map/Unmap agar mousewheel hanya aktif saat frame ini tampil
        self.bind("<Map>",   lambda e: self.bind_all("<MouseWheel>", self._on_mousewheel))
        self.bind("<Unmap>", lambda e: self.unbind_all("<MouseWheel>"))

        self.sf = tk.Frame(self.main_container, bg=CLR["bg"], padx=28, pady=22)
        self.sf.pack(fill="both", expand=True)

    #  HELPERS WIDGET
    def _card(self, parent, title, accent, icon="📋", pady_bottom=15):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="x", pady=(0, pady_bottom))

        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=18, pady=12)
        hd.pack(fill="x")
        tk.Label(hd, text=f"{icon}  {title}", font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent).pack(side="left")
        tk.Frame(outer, bg=accent, height=3).pack(fill="x")

        content = tk.Frame(outer, bg=CLR["surface"], padx=18, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _title_card(self, parent):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["primary"], highlightthickness=1)
        card.pack(fill="x", pady=(0, 18))
        tk.Frame(card, bg=CLR["primary"], height=5).pack(fill="x")
        content = tk.Frame(card, bg=CLR["surface"], padx=20, pady=18)
        content.pack(fill="both", expand=True)
        return content

    def _bullet_row(self, parent, text, badge_color, badge_text="•"):
        f = tk.Frame(parent, bg=CLR["surface"])
        f.pack(anchor="w", pady=4, fill="x")

        badge = tk.Frame(f, bg=badge_color, width=22, height=22)
        badge.pack(side="left", padx=(0, 8))
        badge.pack_propagate(False)
        tk.Label(badge, text=badge_text, font=("Segoe UI", 8, "bold"), fg="white", bg=badge_color).pack(expand=True)

        tk.Label(f, text=text, font=self.body_font, fg=CLR["text_body"], bg=CLR["surface"], wraplength=2000, anchor="w", justify="left"
                 ).pack(side="left", fill="x", expand=True)

    def _kpi_box(self, parent, label, value, fg, bg_color):
        box = tk.Frame(parent, bg=bg_color, highlightbackground=fg, highlightthickness=1, padx=16, pady=12)
        box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(box, text=value, font=("Segoe UI", 22, "bold"), bg=bg_color, fg=fg).pack()
        tk.Label(box, text=label, font=("Segoe UI", 9), bg=bg_color, fg=fg).pack()

    def _warning_box(self, parent, text, level="warning"):
        colors = {
            "warning": (CLR["warning_lt"], CLR["warning"]),
            "danger":  (CLR["danger_lt"],  CLR["danger"]),
            "info":    (CLR["blue_lt"],    CLR["blue"]),
            "success": (CLR["success_lt"], CLR["success"]),
        }
        bg, fg = colors.get(level, colors["warning"])
        box = tk.Frame(parent, bg=bg, highlightbackground=fg, highlightthickness=1)
        box.pack(fill="x", pady=(8, 0))
        tk.Label(box, text=f"{text}", font=("Segoe UI", 9), bg=bg, fg=fg, wraplength=1400, justify="left", padx=12, pady=8).pack(anchor="w")

    #  KONTEN UTAMA
    def _populate(self):
        self._section_judul()
        self._section_masalah_solusi()
        self._section_tujuan()
        self._section_dataset()
        self._section_target()

    # 1. Judul Penelitian
    def _section_judul(self):
        c = self._title_card(self.sf)

        tk.Label(c, text=("ANALISIS PERBANDINGAN KINERJA ALGORITMA NAIVE BAYES DAN K-NEAREST NEIGHBOR (KNN)\nUNTUK KLASIFIKASI DOKUMEN OTOMATIS"),
                 font=("Segoe UI", 13, "bold"), fg=CLR["text_hd"], bg=CLR["surface"], justify="center", anchor="center").pack(fill="x")

        tk.Label(c, text="Studi Kasus: Dokumen Keagenan Pelayaran PT Atika Jaya Samudera Cabang Samarinda",
                 font=("Segoe UI", 10), fg=CLR["text_muted"], bg=CLR["surface"], justify="center").pack(pady=(6, 0))

    # 2. Masalah & Solusi
    def _section_masalah_solusi(self):
        two_col = tk.Frame(self.sf, bg=CLR["bg"])
        two_col.pack(fill="x", pady=(0, 0))

        left  = tk.Frame(two_col, bg=CLR["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(two_col, bg=CLR["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # Masalah
        prob = self._card(left, "Permasalahan", CLR["danger"], "⚠️")
        for p in [
            "Proses pengarsipan dokumen masih manual dan tidak efisien",
            "Klasifikasi dokumen bersifat subjektif dan rentan human error",
            "Belum diketahui algoritma terbaik untuk klasifikasi dokumen otomatis",
        ]:
            self._bullet_row(prob, p, CLR["danger"], "!")

        # Solusi
        sol = self._card(right, "Solusi yang Diusulkan", CLR["purple"], "💡")
        for s in [
            "Digitalisasi dokumen menggunakan OCR untuk mengekstrak teks dari dokumen",
            "Augmentasi data berbasis pola struktural dokumen riil untuk mengatasi keterbatasan data lapangan",
            "Ekstraksi fitur menggunakan TF-IDF untuk representasi dokumen",
            "Perbandingan algoritma Naive Bayes dan KNN dengan evaluasi dua lapis",
        ]:
            self._bullet_row(sol, s, CLR["purple"], "✓")

    # 3. Tujuan
    def _section_tujuan(self):
        c = self._card(self.sf, "Tujuan Penelitian", CLR["blue"], "📌")
        for t in [
            "Membangun sistem klasifikasi dokumen otomatis berbasis teks untuk dokumen keagenan pelayaran PT Atika Jaya Samudera Cabang Samarinda",
            "Menganalisis dan membandingkan performa Naive Bayes dan KNN untuk memperoleh algoritma terbaik",
        ]:
            self._bullet_row(c, t, CLR["blue"], "→")

    # 4. Strategi Dataset & Augmentasi
    def _section_dataset(self):
        c = self._card(self.sf, "Strategi Dataset & Augmentasi Data", CLR["teal"], "🗂️")

        self._warning_box(c,"Keterbatasan lapangan: hanya 15 dokumen riil berhasil dikumpulkan dari PT Atika Jaya Samudera "
            "Cabang Samarinda karena dokumen bersifat rahasia operasional perusahaan.", level="warning")

        tk.Frame(c, bg=CLR["border"], height=1).pack(fill="x", pady=12)

        # Dua kolom: komposisi + strategi evaluasi
        two = tk.Frame(c, bg=CLR["surface"])
        two.pack(fill="x")

        left  = tk.Frame(two, bg=CLR["surface"])
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = tk.Frame(two, bg=CLR["surface"])
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # Komposisi dataset
        tk.Label(left, text="📊  Komposisi Dataset per Kelas", font=("Segoe UI", 10, "bold"), fg=CLR["teal"],bg=CLR["surface"]
                 ).pack(anchor="w", pady=(0, 6))

        hdr = tk.Frame(left, bg=CLR["teal"])
        hdr.pack(fill="x")
        for txt, w in [("Kelas", 14), ("Riil", 7), ("Augmentasi", 12), ("Total", 7)]:
            tk.Label(hdr, text=txt, font=("Segoe UI", 8, "bold"), fg="white", bg=CLR["teal"], width=w, anchor="center", pady=5).pack(side="left")

        kelas_items = [
            ("Sertifikat",  9,  11, CLR["primary"],  CLR["primary_lt"]),
            ("Persetujuan", 3,  17, CLR["success"],  CLR["success_lt"]),
            ("Perbaikan",   2,  18, CLR["warning"],  CLR["warning_lt"]),
            ("Perizinan",   1,  19, CLR["purple"],   CLR["purple_lt"]),
        ]
        for nama, riil, aug, fg, bg in kelas_items:
            row = tk.Frame(left, bg=bg, highlightbackground=fg, highlightthickness=1)
            row.pack(fill="x", pady=(1, 0))
            for txt, w in [(nama, 14), (str(riil), 7), (str(aug), 12), ("20", 7)]:
                tk.Label(row, text=txt, font=("Segoe UI", 9), fg=fg, bg=bg, width=w, anchor="center", pady=6).pack(side="left")

        # Footer total
        tot = tk.Frame(left, bg=CLR["teal"])
        tot.pack(fill="x", pady=(1, 0))
        for txt, w in [("TOTAL", 14), ("15", 7), ("65", 12), ("80", 7)]:
            tk.Label(tot, text=txt, font=("Segoe UI", 9, "bold"), fg="white", bg=CLR["teal"], width=w, anchor="center", pady=6).pack(side="left")

        # Kanan: Metode augmentasi + strategi evaluasi
        tk.Label(right, text="⚙️  Metode Augmentasi Data", font=("Segoe UI", 10, "bold"), fg=CLR["teal"], bg=CLR["surface"]).pack(anchor="w", pady=(0, 6))

        for m in [
            "Parafrase manual mengikuti pola struktural dokumen riil (format surat, field data kapal)",
            "Variasi nama kapal, galangan, tanggal, dan data teknis sesuai domain pelayaran",
            "Cleaning noise OCR (teks cookie/iklan) pada dokumen yang bersumber dari repositori publik",
        ]:
            f = tk.Frame(right, bg=CLR["surface"])
            f.pack(anchor="w", pady=3, fill="x")
            tk.Label(f, text="⚙", font=("Segoe UI", 9), fg=CLR["teal"], bg=CLR["surface"]).pack(side="left", padx=(0, 6))
            tk.Label(f, text=m, font=("Segoe UI", 9), fg=CLR["text_body"], bg=CLR["surface"],
                     wraplength=500, justify="left", anchor="w").pack(side="left", fill="x", expand=True)

        tk.Frame(right, bg=CLR["border"], height=1).pack(fill="x", pady=10)

    def _section_target(self):
        c = self._card(self.sf, "Target Data Sains", CLR["success"], "🎯")

        # Label variabel y
        y_frame = tk.Frame(c, bg=CLR["surface"])
        y_frame.pack(fill="x", pady=(0, 12))
        tk.Label(y_frame, text="Variabel Target (y)  =  Kelas Dokumen", font=("Segoe UI", 11, "bold"),
                 fg=CLR["primary"], bg=CLR["surface"]).pack(anchor="w")

        # Tabel 4 kelas
        kelas_data = [
            ("0", "Sertifikat",  CLR["primary"],  CLR["primary_lt"],
             "Dokumen sertifikat kapal, awak, atau keagenan"),
            ("1", "Persetujuan", CLR["success"],  CLR["success_lt"],
             "Dokumen surat persetujuan dari instansi terkait"),
            ("2", "Perbaikan",   CLR["warning"],  CLR["warning_lt"],
             "Dokumen permintaan atau laporan perbaikan"),
            ("3", "Perizinan",   CLR["purple"],   CLR["purple_lt"],
             "Dokumen izin operasional / masuk-keluar pelabuhan"),
        ]

        # Header tabel
        hdr = tk.Frame(c, bg=CLR["primary"])
        hdr.pack(fill="x")
        for txt, w in [("Nilai y", 8), ("Kelas Dokumen", 18), ("Deskripsi", 55), ("Jumlah Dokumen", 15)]:
            tk.Label(hdr, text=txt, font=("Segoe UI", 9, "bold"), fg="white", bg=CLR["primary"], 
                     width=w, anchor="w", padx=8, pady=6).pack(side="left")

        # Baris per kelas
        for val, nama, fg, bg, deskripsi in kelas_data:
            jumlah = DATASET_INFO["kelas"][nama]["total"]
            row = tk.Frame(c, bg=bg, highlightbackground=fg, highlightthickness=1)
            row.pack(fill="x", pady=(2, 0))

            tk.Label(row, text=val, font=("Segoe UI", 11, "bold"), fg="white", bg=fg, width=8, anchor="center", pady=10).pack(side="left")
            tk.Label(row, text=nama, font=("Segoe UI", 10, "bold"), fg=fg, bg=bg, width=18, anchor="w", padx=8).pack(side="left")
            tk.Label(row, text=deskripsi, font=("Segoe UI", 9), fg=CLR["text_body"], bg=bg, anchor="w", padx=8, wraplength=900
                     ).pack(side="left", fill="x", expand=True)
            tk.Label(row, text=f"{jumlah} dokumen", font=("Segoe UI", 9, "bold"), width=15, anchor="center").pack(side="left")

    #  SCROLL HANDLER
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")