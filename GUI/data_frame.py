import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd
from ocr_extraction import IMAGE_EXTENSIONS
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#import matplotlib.patches as mpatches
#import matplotlib.pyplot as plt

# PALET WARNA
CLR = {
    "bg":           "#F0F2F8", "surface":      "#FFFFFF", "surface2":     "#F8FAFF", "primary":      "#4F46E5", "primary_dk":   "#3730A3",
    "primary_lt":   "#EEF2FF", "success":      "#10B981", "warning":      "#F59E0B", "danger":       "#EF4444", "purple":       "#8B5CF6",
    "blue":         "#3B82F6", "text_hd":      "#1E1B4B", "text_body":    "#374151", "text_muted":   "#6B7280", "border":       "#E5E7EB",
    "synth":        "#F59E0B", "real":         "#10B981",
}

# Data statistik statis
STATIC_STATS = {
    "total":   80,
    "classes": 4,
    "real":    14,
    "synth":   66,
}

SYNTH_SOURCES = [
    ("Website Scribd",    "https://www.scribd.com/home",         "Regulasi pelayaran nasional"),
]

class DataFramePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller
        self._apply_styles()
        self._build_header()

        # Scroll canvas utama
        self.canvas = tk.Canvas(self, bg=CLR["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_content = tk.Frame(self.canvas, bg=CLR["bg"])
        self.scrollable_content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)) 
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # mousewheel scroll
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.main_area = tk.Frame(self.scrollable_content, bg=CLR["bg"], padx=28, pady=22)
        self.main_area.pack(fill="both", expand=True)

        # Blok-blok UI
        self._create_control_card(self.main_area)
        self._build_static_kpi(self.main_area)         # KPI 4-kotak statis
        self._build_table_card(self.main_area)         # Tabel dokumen (dari folder)





    # Style
    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Modern.Treeview", background=CLR["surface"], fieldbackground=CLR["surface"], foreground=CLR["text_body"], rowheight=36,
                         font=("Segoe UI", 12))
        style.configure("Modern.Treeview.Heading", background=CLR["primary_lt"], foreground=CLR["primary"], font=("Segoe UI", 12, "bold"), 
                        relief="flat")
        style.map("Modern.Treeview", background=[("selected", CLR["primary_lt"])], foreground=[("selected", CLR["primary"])])

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Header
    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightthickness=0)
        outer.pack(side="top", fill="x")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(inner, text="Tahap 1: Akuisisi Data", font=("Segoe UI", 18, "bold"), bg=CLR["surface"], fg=CLR["text_hd"]
                 ).pack(side="left", anchor="w")

    # Card builder
    def _make_card(self, parent, title, pady_bottom=20):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightthickness=0)
        outer.pack(fill="x", pady=(0, pady_bottom))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=CLR["primary"]).pack(side="left")
        tk.Frame(outer, bg=CLR["primary"], height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _make_warning_card(self, parent, title, pady_bottom=20):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightthickness=0)
        outer.pack(fill="x", pady=(0, pady_bottom))
        hd = tk.Frame(outer, bg="#FFFBEB", padx=15, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=f"⚠  {title}", font=("Segoe UI", 11, "bold"), bg="#FFFBEB", fg=CLR["warning"]).pack(side="left")
        tk.Frame(outer, bg=CLR["warning"], height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        content.pack(fill="both", expand=True)
        return content

    # Panel kontrol
    def _create_control_card(self, parent):
        card = tk.Frame(parent, bg=CLR["surface"], highlightthickness=0)
        card.pack(fill="x", pady=(0, 20))
        tk.Frame(card, bg=CLR["primary"], height=4).pack(fill="x")
        content = tk.Frame(card, bg=CLR["surface"], padx=20, pady=20)
        content.pack(fill="both", expand=True)
        tk.Button(content, text="+ Pilih Folder Dataset", command=self.load_metadata, bg=CLR["primary"], fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=20, pady=10, cursor="hand2").pack(side="left")
        self.info_folder = tk.Label(content, text="Status: Menunggu folder...", bg=CLR["surface"], fg=CLR["text_muted"])
        self.info_folder.pack(side="left", padx=20)

    # KPI statis
    def _build_static_kpi(self, parent):
        card = self._make_card(parent, "Ringkasan Dataset")
        kpis = [
            ("Total Dokumen",  str(STATIC_STATS["total"]),   CLR["primary"],  CLR["primary_lt"]),
            ("Jumlah Kelas",   str(STATIC_STATS["classes"]), CLR["success"],  "#F0FDF4"),
            ("Data Riil",      str(STATIC_STATS["real"]),    CLR["real"],     "#F0FDF4"),
            ("Data Sintetis",  str(STATIC_STATS["synth"]),   CLR["warning"],  "#FFFBEB"),
        ]
        row = tk.Frame(card, bg=CLR["surface"])
        row.pack(fill="x")
        for title, val, fg, bg in kpis:
            kf = tk.Frame(row, bg=bg, highlightthickness=1, highlightbackground=fg, padx=20, pady=18)
            kf.pack(side="left", fill="both", expand=True, padx=(0, 10))
            tk.Label(kf, text=val, font=("Segoe UI", 22, "bold"), bg=bg, fg=fg).pack()
            tk.Label(kf, text=title, font=("Segoe UI", 12), bg=bg, fg=fg).pack()

    # Tabel dokumen
    def _build_table_card(self, parent):
        card = self._make_card(parent, "Daftar Dokumen (dari Folder yang Dipilih)")
        cols = ("filename", "label", "ext", "file_type")
        self.tree = ttk.Treeview(card, columns=cols, show="headings", height=10, style="Modern.Treeview")
        self.tree.heading("filename",  text="Nama File")
        self.tree.heading("label",     text="Kategori")
        self.tree.heading("ext",       text="Format")
        self.tree.heading("file_type", text="Jenis")
        self.tree.column("label",     width=150, anchor="center")
        self.tree.column("ext",       width=100, anchor="center")
        self.tree.column("file_type", width=100, anchor="center")

        vsb = ttk.Scrollbar(card, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="x", pady=5)

        self.tree.tag_configure("Persetujuan", background="#EEF2FF", foreground=CLR["primary"])
        self.tree.tag_configure("Sertifikat",  background="#F0FDF4", foreground=CLR["success"])
        self.tree.tag_configure("Perbaikan",   background="#FFFBEB", foreground=CLR["warning"])
        self.tree.tag_configure("Perizinan",   background="#F5F3FF", foreground=CLR["purple"])

    def infer_source_type(self, file_path):
        p = file_path.lower()
        if any(k in p for k in ("sintetis", "synthetic", "augmentasi", "augment")):
            return "synthetic"
        if any(k in p for k in ("riil", "real", "pt_atika", "atika")):
            return "real"
        return "unknown"

    # Load dari folder
    def load_metadata(self):
        path = filedialog.askdirectory()
        if not path:
            return
        data        = []
        skipped     = []
        valid_exts  = {".pdf", ".docx"} | IMAGE_EXTENSIONS

        for label in os.listdir(path):
            l_path = os.path.join(path, label)
            if not os.path.isdir(l_path):
                continue
            for f in os.listdir(l_path):
                ext = os.path.splitext(f)[1].lower()
                if ext not in valid_exts:
                    skipped.append(f)
                    continue

                file_path  = os.path.join(l_path, f)
                is_img     = ext in IMAGE_EXTENSIONS
                file_type  = "Gambar" if is_img else ext.lstrip(".").upper()

                data.append({
                    "filename":    f,
                    "path":        file_path,
                    "label":       label,
                    "ext":         ext,
                    "file_type":   file_type,
                    "text_raw":    None,
                    "source_type": self.infer_source_type(file_path),
                    "ocr_error":   "",
                })

        self.controller.df = pd.DataFrame(data)

        if self.controller.df.empty:
            msg = "Tidak ada file PDF, DOCX, atau gambar yang ditemukan.\n"
            if skipped:
                msg += f"\nFile diabaikan ({len(skipped)} file): {', '.join(skipped[:5])}"
                if len(skipped) > 5:
                    msg += f" ... dan {len(skipped) - 5} lainnya."
            messagebox.showwarning("Folder Kosong", msg)
            self.controller.df = None
            return

        self._refresh_treeview()
        n   = len(self.controller.df)
        msg = f"{n} dokumen ditemukan di: {path}"
        if skipped:
            msg += f"  |  {len(skipped)} file diabaikan (format tidak didukung)"
        self.info_folder.config(text=msg, fg=CLR["success"])

    def _refresh_treeview(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for _, row in self.controller.df.iterrows():
            self.tree.insert(
                "", "end",
                values=(
                    row["filename"],
                    row["label"],
                    row["ext"].upper(),
                    row.get("file_type", "—"),
                ),
                tags=(row["label"],)
            )