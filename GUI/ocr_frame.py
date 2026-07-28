import tkinter as tk
from tkinter import ttk, messagebox
import threading
import re
import os
import fitz
from PIL import Image, ImageTk
from collections import Counter
from ocr_extraction import extract_text, is_scanned_text, IMAGE_EXTENSIONS
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import numpy as np

# ── Palet warna global ──────────────────────────────────────────────────────
CLR = {
    "bg":           "#F0F2F8", "surface":      "#FFFFFF", "surface2":     "#F8FAFF", "primary":      "#4F46E5",
    "primary_dk":   "#3730A3", "primary_lt":   "#EEF2FF", "success":      "#10B981", "warning":      "#F59E0B",
    "danger":       "#EF4444", "purple":       "#8B5CF6", "blue":         "#3B82F6", "text_hd":      "#1E1B4B",
    "text_body":    "#374151", "text_muted":   "#6B7280", "border":       "#E5E7EB", "teal":         "#14B8A6",
    "orange":       "#F97316",
}

# Ambang batas noise dan panjang teks
NOISE_THRESHOLD_GOOD   = 5.0
NOISE_THRESHOLD_MEDIUM = 15.0
CHAR_THRESHOLD_FAIL    = 100

class OCRFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller
        self._selected_row_path = None 

        self._apply_styles()
        self._build_header()

        # Scrollable Area
        self.canvas    = tk.Canvas(self, bg=CLR["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_content = tk.Frame(self.canvas, bg=CLR["bg"])

        self.scrollable_content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Main Area
        self.main_area = tk.Frame(self.scrollable_content, bg=CLR["bg"], padx=28, pady=22)
        self.main_area.pack(fill="both", expand=True)

        # 1. Kontrol OCR
        self.control_panel = self._create_control_card(self.main_area, CLR["primary"])
        self._setup_control_ui(self.control_panel)

        # 2. KPI cards (diisi setelah OCR)
        self.kpi_area = tk.Frame(self.main_area, bg=CLR["bg"])
        self.kpi_area.pack(fill="x", pady=(0, 10))

        # 3. Tabel kualitas OCR per dokumen (dengan flag merah)
        self.table_panel = self.create_card(self.main_area, "Kualitas OCR per Dokumen", CLR["primary"])
        self._setup_treeview_ui(self.table_panel)

        # 4. Preview panel (gambar asli ↔ teks OCR)
        self._build_preview_panel(self.main_area)

        # 5. Container EDA grafik
        self.eda_container = tk.Frame(self.main_area, bg=CLR["bg"])
        self.eda_container.pack(fill="x", pady=(5, 10))

    # Style
    def _apply_styles(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TProgressbar", thickness=12, troughcolor=CLR["bg"], background=CLR["primary"], borderwidth=0)
        s.configure("Modern.Treeview", background=CLR["surface"], fieldbackground=CLR["surface"], foreground=CLR["text_body"], rowheight=36, font=("Segoe UI", 12))
        s.configure("Modern.Treeview.Heading", background=CLR["primary_lt"], foreground=CLR["primary"], font=("Segoe UI", 12, "bold"))
        s.map("Modern.Treeview", background=[("selected", CLR["primary_lt"])], foreground=[("selected", CLR["primary"])])

    # Scroll
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Header
    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(side="top", fill="x")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(inner, text="Tahap 2: Ekstraksi Teks & EDA Kualitas OCR", font=("Segoe UI", 18, "bold"),
                 bg=CLR["surface"], fg=CLR["text_hd"]).pack(side="left", anchor="w")

    # Card 
    def create_card(self, parent, title, accent_color, expand=False):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="both" if expand else "x", expand=expand, pady=(0, 20))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def create_card_for_grid(self, parent, title, accent_color, row, col, padx):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.grid(row=row, column=col, sticky="nsew", padx=padx, pady=(0, 10))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=8)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 12, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=10, pady=10)
        content.pack(fill="both", expand=True)
        return content

    def _create_control_card(self, parent, accent_color):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        card.pack(fill="x", pady=(0, 20))
        tk.Frame(card, bg=accent_color, height=4).pack(fill="x")
        content = tk.Frame(card, bg=CLR["surface"], padx=20, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _setup_control_ui(self, parent):
        self.btn_ocr = tk.Button(parent, text="Jalankan OCR untuk Semua Dokumen", command=self.start_ocr, bg=CLR["primary"], 
                                 fg="white", font=("Segoe UI", 10, "bold"), relief="flat", pady=12, cursor="hand2")
        self.btn_ocr.pack(fill="x", pady=(0, 10))
        self.progress = ttk.Progressbar(parent, mode="determinate", style="TProgressbar")
        self.progress.pack(fill="x")
        self.lbl_status = tk.Label(parent, text="Siap mengekstrak teks...", bg=CLR["surface"], fg=CLR["text_muted"])
        self.lbl_status.pack(pady=5)

    # Tabel Kualitas OCR per Dokumen
    def _setup_treeview_ui(self, parent):
        # Legend flag
        legend = tk.Frame(parent, bg=CLR["surface"])
        legend.pack(fill="x", pady=(0, 8))

        cols = ("filename", "label", "metode", "chars", "words", "noise", "flag")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=12, style="Modern.Treeview")

        self.tree.heading("filename", text="Nama File")
        self.tree.heading("label",    text="Kategori")
        self.tree.heading("metode",   text="Metode")
        self.tree.heading("chars",    text="Karakter")
        self.tree.heading("words",    text="Kata")
        self.tree.heading("noise",    text="Noise %")
        self.tree.heading("flag",     text="Status")

        self.tree.column("filename", width=260)
        self.tree.column("label",    width=100, anchor="center")
        self.tree.column("metode",   width=80,  anchor="center")
        self.tree.column("chars",    width=90,  anchor="center")
        self.tree.column("words",    width=80,  anchor="center")
        self.tree.column("noise",    width=90,  anchor="center")
        self.tree.column("flag",     width=140, anchor="center")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 5), pady=5)
        self.tree.pack(fill="x", padx=5, pady=5)

        # Tag warna baris
        self.tree.tag_configure("quality_good",   foreground=CLR["success"], background="#F0FDF4")
        self.tree.tag_configure("quality_medium", foreground=CLR["warning"], background="#FFFBEB")
        self.tree.tag_configure("quality_bad",    foreground=CLR["danger"], background="#FEF2F2")
        self.tree.tag_configure("ocr_fail",       foreground="#DC2626", background="#FEE2E2", font=("Segoe UI", 12, "bold"))

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._row_data = {}

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        entry = self._row_data.get(iid)
        if entry:
            self._update_preview(entry["path"], entry["text"], entry["metode"])

    # Preview Gambar Dokumen ↔ Teks OCR
    def _build_preview_panel(self, parent):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="x", pady=(0, 20))

        hd = tk.Frame(outer, bg="#F5F3FF", padx=15, pady=10)
        hd.pack(fill="x")
        hd_left = tk.Frame(hd, bg="#F5F3FF")
        hd_left.pack(side="left", fill="y")
        tk.Label(hd_left, text="Preview Dokumen", font=("Segoe UI", 11, "bold"), bg="#F5F3FF", fg=CLR["purple"]
                 ).pack(side="left")
        tk.Frame(outer, bg=CLR["purple"], height=3).pack(fill="x")

        body = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Panel kiri: render halaman dokumen
        left = tk.Frame(body, bg=CLR["surface2"], highlightthickness=1, highlightbackground=CLR["border"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Frame(left, bg=CLR["purple"], height=3).pack(fill="x")
        tk.Label(left, text="📄  Halaman Pertama Dokumen", font=("Segoe UI", 12, "bold"), bg=CLR["surface2"], fg=CLR["purple"],
                 padx=10, pady=6).pack(anchor="w")

        self.preview_canvas = tk.Canvas(left, bg=CLR["surface2"], width=440, height=580, highlightthickness=0)
        self.preview_canvas.pack(padx=10, pady=(0, 10))
        self._preview_img_ref = None

        self.lbl_preview_info = tk.Label(left, text="Belum ada dokumen dipilih.", font=("Segoe UI", 8), bg=CLR["surface2"],
                                         fg=CLR["text_muted"], wraplength=420, justify="left", padx=10)
        self.lbl_preview_info.pack(anchor="w", pady=(0, 8))

        # Panel kanan: teks hasil OCR
        right = tk.Frame(body, bg=CLR["surface2"], highlightthickness=1, highlightbackground=CLR["border"])
        right.grid(row=0, column=1, sticky="nsew")

        tk.Frame(right, bg=CLR["success"], height=3).pack(fill="x")
        tk.Label(right, text="Teks Hasil Ekstraksi", font=("Segoe UI", 12, "bold"), bg=CLR["surface2"], fg=CLR["success"],
                 padx=10, pady=6).pack(anchor="w")

        txt_frame = tk.Frame(right, bg=CLR["surface2"])
        txt_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        txt_vsb = ttk.Scrollbar(txt_frame, orient="vertical")
        txt_vsb.pack(side="right", fill="y")
        self.txt_ocr = tk.Text(txt_frame, bg=CLR["surface"], fg=CLR["text_body"], font=("Courier New", 12), relief="flat", wrap="word",
                               yscrollcommand=txt_vsb.set, height=30, padx=10, pady=10, state="disabled")
        self.txt_ocr.pack(side="left", fill="both", expand=True)
        txt_vsb.config(command=self.txt_ocr.yview)

        # Metrik ringkasan dokumen yang dipilih
        self.lbl_ocr_stats = tk.Label(right, text="Karakter: —   |   Kata: —   |   Metode: —   |   Noise: —", 
                                      font=("Segoe UI", 12, "bold"), bg=CLR["surface2"], fg=CLR["text_muted"], padx=10, pady=6)
        self.lbl_ocr_stats.pack(anchor="w")

    def _update_preview(self, file_path, ocr_text, metode="Digital"):
        # Teks panel
        self.txt_ocr.config(state="normal")
        self.txt_ocr.delete("1.0", "end")
        display_text = ocr_text if ocr_text and ocr_text.strip() else "(Tidak ada teks terdeteksi)"
        self.txt_ocr.insert("1.0", display_text)
        self.txt_ocr.config(state="disabled")

        char_cnt  = len(ocr_text) if ocr_text else 0
        word_cnt  = len(ocr_text.split()) if ocr_text else 0
        noise_lst = re.findall(r"[^a-zA-Z0-9\s.,?!\u00C0-\u024F]", ocr_text or "")
        noise_pct = len(noise_lst) / len(ocr_text) * 100 if char_cnt > 0 else 0
        # metode sudah disimpan di self_row_data, sehingga tidak perlu ditebak ulang

        flag_txt = ""
        if char_cnt < CHAR_THRESHOLD_FAIL:
            flag_txt = "KEMUNGKINAN GAGAL OCR"
        self.lbl_ocr_stats.config(text=f"Karakter: {char_cnt:,}   |   Kata: {word_cnt:,}   |   "
                                   f"Metode: {metode}   |   Noise: {noise_pct:.1f}%{flag_txt}",
                                   fg=CLR["danger"] if char_cnt < CHAR_THRESHOLD_FAIL else CLR["text_muted"]
        )

        # Gambar panel
        self.preview_canvas.delete("all")
        self._preview_img_ref = None
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                with fitz.open(file_path) as doc:
                    if len(doc) == 0: raise ValueError("PDF kosong")
                    n_pages = len(doc)
                    page = doc[0]
                    pix  = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                    img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    cw, ch = 440, 580
                    img.thumbnail((cw, ch), Image.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img)
                    self._preview_img_ref = tk_img
                    self.preview_canvas.config(width=cw, height=ch)
                    self.preview_canvas.create_image(
                        cw // 2, ch // 2, anchor="center", image=tk_img)
                info = (f"File: {os.path.basename(file_path)}\n"
                        f"Halaman: {n_pages} halaman  |  "
                        f"Jenis: PDF  |  Resolusi render: 1.5×")

            elif ext in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(file_path)
                    # GIF animasi: ambil frame pertama
                    if getattr(img, "is_animated", False):
                        img.seek(0)
                    if img.mode not in ("RGB", "RGBA", "L"):
                        img = img.convert("RGB")
                    cw, ch = 440, 580
                    img_disp = img.copy()
                    img_disp.thumbnail((cw, ch), Image.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img_disp)
                    self._preview_img_ref = tk_img
                    self.preview_canvas.config(width=cw, height=ch)
                    self.preview_canvas.create_image(
                        cw // 2, ch // 2, anchor="center", image=tk_img)
                    info = (
                        f"File: {os.path.basename(file_path)}\n"
                        f"Jenis: Gambar ({ext.upper().lstrip('.')})  |  "
                        f"Ukuran asli: {img.size[0]}×{img.size[1]} px  |  "
                        f"Mode: {img.mode}"
                    )
                except Exception as img_err:
                    self.preview_canvas.config(width=440, height=580)
                    self.preview_canvas.create_rectangle(20, 20, 420, 560, fill="#FEF2F2",
                                                         outline=CLR["danger"], width=2)
                    self.preview_canvas.create_text(220, 260, text="🖼", font=("Segoe UI", 64),
                                                    fill=CLR["danger"])
                    self.preview_canvas.create_text(220, 340, text=f"Gagal memuat gambar:\n{img_err}",
                                                    font=("Segoe UI", 9), fill=CLR["danger"],
                                                    justify="center", width=380)
                    info = f"File: {os.path.basename(file_path)}\nError: {img_err}"

            elif ext == ".docx":
                self.preview_canvas.config(width=440, height=580)
                self.preview_canvas.create_rectangle(20, 20, 420, 560, fill="#EEF2FF", outline=CLR["border"], width=2)
                self.preview_canvas.create_text(220, 260, text="📄", font=("Segoe UI", 64), fill=CLR["primary"])
                self.preview_canvas.create_text(220, 330, text="Dokumen DOCX", font=("Segoe UI", 14, "bold"), fill=CLR["primary"])
                self.preview_canvas.create_text(220, 360, text="(Preview gambar tidak tersedia untuk DOCX)",
                                                font=("Segoe UI", 9), fill=CLR["text_muted"])
                info = (f"File: {os.path.basename(file_path)}\n" f"Jenis: DOCX  |  Preview visual tidak tersedia.")

            else:
                self.preview_canvas.config(width=440, height=580)
                self.preview_canvas.create_rectangle(20, 20, 420, 560, fill="#F3F4F6",
                                                     outline=CLR["border"], width=2)
                self.preview_canvas.create_text(220, 290,
                                                text=f"⚠ Format tidak didukung\n'{ext}'",
                                                font=("Segoe UI", 12), fill=CLR["text_muted"],
                                                justify="center")
                info = f"Format tidak dikenali: {ext}"

            self.lbl_preview_info.config(text=info)

        except Exception as e:
            self.preview_canvas.create_text(220, 290, text=f"⚠ Gagal render:\n{e}", 
                                            font=("Segoe UI", 9), fill=CLR["danger"], justify="center")
            self.lbl_preview_info.config(text=str(e))

    # Logika OCR
    def start_ocr(self):
        if self.controller.df is None or self.controller.df.empty:
            return messagebox.showwarning("Peringatan", "Metadata belum dimuat.")
        self.btn_ocr.config(state="disabled")
        for i in self.tree.get_children():
            self.tree.delete(i)
        self._row_data.clear()
        threading.Thread(target=self.ocr_worker, daemon=True).start()

    def ocr_worker(self):
        df    = self.controller.df
        total = len(df)
        self.after(0, lambda: self.progress.configure(maximum=total, value=0))

        try:
            for i, row in df.iterrows():
                text, metode = extract_text(row["path"])

                noise_chars = re.findall(r"[^a-zA-Z0-9\s.,?!\u00C0-\u024F]", text)
                noise_score = (len(noise_chars) / len(text) * 100) if len(text) > 0 else 0
                word_count  = len(text.split()) if text.strip() else 0

                self.after(0, lambda idx=i, t=text, ns=noise_score, wc=word_count, m=metode:
                           self._apply_ocr_result(idx, t, ns, wc, m))
                self.after(0, lambda r=row, t=text, ns=noise_score, wc=word_count, m=metode, cnt=i + 1:
                           self._update_row_ui(r, t, m, ns, wc, cnt, total))

        except Exception as e:
            self.after(0, lambda err=e: messagebox.showerror(
                "Error OCR", f"Proses OCR terhenti:\n{err}"))
        finally:
            self.after(0, self.render_eda_summary)

    def _apply_ocr_result(self, idx, text, noise_score, word_count, metode):
        self.controller.df.at[idx, "text_raw"]    = text
        self.controller.df.at[idx, "noise_score"] = noise_score
        self.controller.df.at[idx, "word_count"]  = word_count
        self.controller.df.at[idx, "metode_ocr"]  = metode

    def _update_row_ui(self, row, text, metode, noise_score, word_count, cnt, total):
        fname     = row["filename"]
        label_cat = row.get("label", "—")
        char_cnt  = len(text)
        path      = row["path"]

        # Label status hasil OCR berdasarkan ambang batas karakter dan noise score
        if metode in ("OCR-Error", "Digital-Error", "Unsupported"):
            status = f"Error: {metode}"
            tag    = "ocr_fail"
        elif char_cnt < CHAR_THRESHOLD_FAIL:
            status = "Teknik OCR gagal menghasilkan teks?"
            tag    = "ocr_fail"
        elif noise_score < NOISE_THRESHOLD_GOOD:
            status = "Baik"
            tag    = "quality_good"
        elif noise_score < NOISE_THRESHOLD_MEDIUM:
            status = "Sedang"
            tag    = "quality_medium"
        else:
            status = "Buruk"
            tag    = "quality_bad"

        iid = self.tree.insert(
            "", "end",
            values=(fname, label_cat, metode, f"{char_cnt:,}", word_count, f"{noise_score:.1f}%", status),
            tags=(tag,))

        # Metode simpan data, sehingga tidak perlu melakukan OCR ulang ketika baris dipilih untuk preview
        self._row_data[iid] = {"path": path, "text": text, "metode": metode}

        self.progress["value"] = cnt
        self.lbl_status.config(text=f"Progres: {cnt} / {total}")

    # Exploratory Data Analisys untuk kualitas hasil ekstraksi OCR
    def render_eda_summary(self):
        self.btn_ocr.config(state="normal")
        for w in self.kpi_area.winfo_children():
            w.destroy()
        for w in self.eda_container.winfo_children():
            w.destroy()

        df = self.controller.df
        for col, default in [("noise_score", 0.0), ("word_count", 0), ("text_raw", ""), ("metode_ocr", "Digital")]:
            if col not in df.columns:
                df[col] = default
        df["text_raw"]   = df["text_raw"].fillna("")
        df["char_count"] = df["text_raw"].apply(len)

        total_docs  = len(df)
        empty_docs  = df[df["text_raw"].str.strip() == ""].shape[0]
        avg_noise   = df["noise_score"].mean()
        n_ocr       = (df["metode_ocr"] == "OCR").sum()
        n_digital   = (df["metode_ocr"] == "Digital").sum()
        good_docs   = (df["noise_score"] <  NOISE_THRESHOLD_GOOD).sum()
        medium_docs = ((df["noise_score"] >= NOISE_THRESHOLD_GOOD) & (df["noise_score"] <  NOISE_THRESHOLD_MEDIUM)).sum()
        bad_docs    = (df["noise_score"] >= NOISE_THRESHOLD_MEDIUM).sum()
        fail_docs   = (df["char_count"]  <  CHAR_THRESHOLD_FAIL).sum()

        self._render_kpi_bar(total_docs, empty_docs, avg_noise, n_ocr, n_digital, good_docs, medium_docs, bad_docs, fail_docs)

        self.eda_container.columnconfigure(0, weight=1, uniform="eda_group")
        self.eda_container.columnconfigure(1, weight=1, uniform="eda_group")

        self._chart_metode_pie(df, n_digital, n_ocr)
        self._chart_char_length(df)
        self._chart_top_noise_chars(df)

        self.scrollable_content.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        messagebox.showinfo("Sukses", "Ekstraksi dan Analisis Kualitas Selesai!")

    # KPI bar
    def _render_kpi_bar(self, total, empty, avg_noise,
                        n_ocr, n_digital, good, medium, bad, fail=0):
        frame = tk.Frame(self.kpi_area, bg=CLR["bg"])
        frame.pack(fill="x")
        kpis = [
            ("Baik",    str(good),    CLR["success"]),
            ("Sedang",  str(medium),  CLR["warning"]),
            ("Buruk/Gagal",       str(fail),    "#DC2626"),
        ]
        for col_idx, (label, value, color) in enumerate(kpis):
            frame.columnconfigure(col_idx, weight=1)
            card = tk.Frame(frame, bg=CLR["surface"], highlightbackground=color, highlightthickness=2)
            card.grid(row=0, column=col_idx, sticky="ew", padx=(0 if col_idx == 0 else 6, 0), pady=(0, 10))
            tk.Frame(card, bg=color, height=4).pack(fill="x")
            tk.Label(card, text=value, font=("Segoe UI", 22, "bold"), bg=CLR["surface"], fg=color).pack(pady=(8, 0))
            tk.Label(card, text=label, font=("Segoe UI", 8), bg=CLR["surface"], fg=CLR["text_muted"], 
                     wraplength=100).pack(pady=(0, 8))

    # Pie Metode
    def _chart_metode_pie(self, df, n_digital, n_ocr):
        c = self.create_card_for_grid(
            self.eda_container,
            "Proporsi Dokumen Digital dan Dokumen Hasil Scan",
            CLR["blue"], 0, 0, (0, 8))
        fig = Figure(figsize=(5, 3.2), dpi=90)
        fig.patch.set_facecolor(CLR["surface"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(CLR["surface"])
        labels, sizes, colors = [], [], []
        if n_digital > 0:
            labels.append(f"Digital\n({n_digital} file)")
            sizes.append(n_digital)
            colors.append(CLR["blue"])
        if n_ocr > 0:
            labels.append(f"OCR (Scan)\n({n_ocr} file)")
            sizes.append(n_ocr)
            colors.append(CLR["purple"])
        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 9})
            for at in autotexts:
                at.set_fontweight("bold")
                at.set_color("white")
        else:
            ax.text(0.5, 0.5, "Tidak ada data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Dokumen Digital lebih cepat & akurat\ndibanding OCR (scan)",
                     fontsize=8, color=CLR["text_muted"], pad=6)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="both", expand=True)

    # Grafik Distribusi Panjang Teks (Karakter)
    def _chart_char_length(self, df):
        c = self.create_card_for_grid(
            self.eda_container,
            "Sebaran Panjang Teks (Karakter per Dokumen)",
            CLR["warning"], 0, 1, (0, 8))

        fig = Figure(figsize=(5, 3.2), dpi=90)
        fig.patch.set_facecolor(CLR["surface"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(CLR["surface2"])

        char_counts = df["char_count"].dropna().values
        if len(char_counts) > 0:
            bins = min(20, len(char_counts))
            ax.hist(char_counts, bins=bins, color=CLR["primary"], edgecolor="white", linewidth=0.8, alpha=0.85)

        ax.set_xlabel("Jumlah Karakter", fontsize=8, color=CLR["text_muted"])
        ax.set_ylabel("Jumlah Dokumen",  fontsize=8, color=CLR["text_muted"])
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Panjang Teks per Dokumen",
                     fontsize=8, color=CLR["text_muted"], pad=6)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="both", expand=True)

    # Top-10 Karakter Noise
    def _chart_top_noise_chars(self, df):
        outer = tk.Frame(self.eda_container, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 10))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=8)
        hd.pack(fill="x")
        tk.Label(hd, text="E  |  Top-10 Karakter Noise Terdeteksi (seluruh corpus)", font=("Segoe UI", 10, "bold"),
                 bg=CLR["primary_lt"], fg=CLR["danger"]).pack(side="left")
        tk.Frame(outer, bg=CLR["danger"], height=3).pack(fill="x")
        c = tk.Frame(outer, bg=CLR["surface"], padx=10, pady=10)
        c.pack(fill="both", expand=True)

        all_text    = "".join(df["text_raw"].tolist())
        noise_chars = re.findall(r"[^a-zA-Z0-9\s.,?!\u00C0-\u024F]", all_text)
        top_noise   = Counter(noise_chars).most_common(10)

        fig = Figure(figsize=(10, 2.5), dpi=90)
        fig.patch.set_facecolor(CLR["surface"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(CLR["surface2"])

        if top_noise:
            chars, counts = zip(*top_noise)
            norm_counts = [ct / max(counts) for ct in counts]
            bar_colors  = [
                "#{:02X}{:02X}{:02X}".format(
                    int(79 + (239 - 79) * (1 - n)),
                    int(70 + (68  - 70) * (1 - n)),
                    int(229 + (68 - 229) * (1 - n))
                )
                for n in norm_counts
            ]
            bars = ax.barh(chars, counts, color=bar_colors,
                           edgecolor="white", linewidth=0.8)
            for bar, val in zip(bars, counts):
                ax.text(bar.get_width() + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,}", va="center", fontsize=8,
                        color=CLR["text_body"], fontweight="bold")
            ax.invert_yaxis()
            ax.set_xlabel("Frekuensi kemunculan", fontsize=8, color=CLR["text_muted"])
        else:
            ax.text(0.5, 0.5, "Tidak ada karakter noise — hasil OCR sangat bersih!",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color=CLR["success"])
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("10 Noise Terbanyak dalam Korpus",
                     fontsize=8, color=CLR["text_muted"], pad=6)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="both", expand=True)