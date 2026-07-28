import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import numpy as np
from preprocessing import (
    case_folding, cleaning, tokenization,
    stopword_removal, normalization, stemming,
    proper_noun_removal, preprocess_text,
    NORMALIZATION_DICT, custom_stopwords, MARITIME_WHITELIST,
)

# ── Palet warna ──────────────────────────────────────────────────────────────
CLR = {
    "bg":           "#F0F2F8", "surface":      "#FFFFFF", "surface2":     "#F8FAFF", "primary":      "#4F46E5",
    "primary_dk":   "#3730A3", "primary_lt":   "#EEF2FF", "success":      "#10B981", "warning":      "#F59E0B",
    "danger":       "#EF4444", "purple":       "#8B5CF6", "blue":         "#3B82F6", "text_hd":      "#1E1B4B",
    "text_body":    "#374151", "text_muted":   "#6B7280", "border":       "#E5E7EB", "teal":         "#14B8A6",
    "orange":       "#F97316",
}

# Warna berbeda per tahap pipeline
STEP_COLORS = {
    "Raw":   "#94A3B8", "CF":    CLR["blue"], "Clean": CLR["warning"], "Tok":   CLR["purple"], "Norm":  CLR["teal"],
    "SW":    CLR["orange"], "PN":    CLR["danger"], "Stem":  CLR["success"],
}

STEPS_META = [
    # (key, tab_label, penjelasan_singkat)
    ("Raw",  "Asli",           "Teks mentah dari hasil OCR / ekstraksi dokumen."),
    ("CF",   "1. Case Fold",   "Semua huruf diubah ke huruf kecil agar token 'Kapal' == 'kapal'."),
    ("Clean","2. Cleaning",    "Hapus URL, email, angka, tanda baca, karakter CamScanner, base64, cookie banner."),
    ("Tok",  "3. Tokenisasi",  "Pisahkan teks menjadi daftar token (kata) berdasarkan spasi."),
    ("Norm", "4. Normalisasi", "Ganti singkatan → kata lengkap (kpd→kepada, yg→yang, dll) sesuai NORMALIZATION_DICT."),
    ("SW",   "5. Stopword",    "Hapus kata umum (dari Sastrawi + custom maritim: ttd, acc, yth, dll) dan token < 3 karakter."),
    ("PN",   "6. Proper Noun", "Hapus nama kapal/orang berdasarkan heuristik kapital & prefix. Kata di MARITIME_WHITELIST tetap dipertahankan."),
    ("Stem", "7. Stemming",    "Reduksi kata ke bentuk dasar menggunakan Sastrawi Stemmer (algoritma Enhanced Confix Stripping untuk Bahasa Indonesia)."),
]

class PreprocessFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller

        self._apply_styles()
        self._build_header()

        # Control strip
        self.fixed_ctrl = tk.Frame(self, bg=CLR["bg"], padx=28, pady=10)
        self.fixed_ctrl.pack(fill="x", side="top")
        self.card_control = self._create_control_card(self.fixed_ctrl, CLR["primary"])
        self.card_control.pack(fill="x", expand=True)
        self._setup_control_ui(self.card_control)

        # Scrollable area
        self.scroll_container = tk.Frame(self, bg=CLR["bg"])
        self.scroll_container.pack(fill="both", expand=True)

        self.canvas    = tk.Canvas(self.scroll_container, bg=CLR["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.scroll_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=CLR["bg"], padx=28, pady=10)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.scroll_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        # Blok konten
        self._build_trace_panel(self.scroll_frame)
        self._build_eda_panel(self.scroll_frame)

    # Styles
    def _apply_styles(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TProgressbar", thickness=10, troughcolor=CLR["bg"], background=CLR["primary"], borderwidth=0)
        s.configure("Modern.Treeview", background=CLR["surface"], fieldbackground=CLR["surface"], rowheight=36, 
                    font=("Segoe UI", 12))
        s.configure("Modern.Treeview.Heading", background=CLR["primary_lt"], foreground=CLR["primary"], 
                    font=("Segoe UI", 12, "bold"))
        s.map("Modern.Treeview", background=[("selected", CLR["primary_lt"])], foreground=[("selected", CLR["primary"])])
        s.configure("Step.TNotebook.Tab", font=("Segoe UI", 12, "bold"), padding=[10, 5])

    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="x", side="top")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(inner, text="Tahap 3: Pra-pemrosesan Teks & EDA Pra-pemrosesan", font=("Segoe UI", 18, "bold"),
                 bg=CLR["surface"], fg=CLR["text_hd"]).pack(side="left", anchor="w")

    def _create_control_card(self, parent, accent_color):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        card.pack(fill="x")
        tk.Frame(card, bg=accent_color, height=4).pack(fill="x")
        inner = tk.Frame(card, bg=CLR["surface"], padx=20, pady=15)
        inner.pack(fill="x", expand=True)
        return inner

    def create_card(self, parent, title, accent_color, pady_bottom=20):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="x", pady=(0, pady_bottom))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=18, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=15)
        content.pack(fill="both", expand=True)
        return content

    # Control Panel
    def _setup_control_ui(self, parent):
        row1 = tk.Frame(parent, bg=CLR["surface"])
        row1.pack(fill="x", expand=True)

        self.btn_run = tk.Button(
            row1, text="Jalankan Preprocessing untuk Semua Dokumen",command=self.run_preprocess, bg=CLR["primary"], 
            fg="white", font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", padx=20, pady=8)
        self.btn_run.pack(side="left")

        # Dropdown pilih dokumen
        self.nav_frame = tk.Frame(row1, bg=CLR["surface2"], padx=10, pady=5, highlightbackground=CLR["border"], 
                                  highlightthickness=1)
        tk.Label(self.nav_frame, text="Trace Dokumen:", bg=CLR["surface2"], font=("Segoe UI", 12, "bold"),
                 fg=CLR["text_body"]).pack(side="left", padx=5)
        self.combo_doc = ttk.Combobox(self.nav_frame, width=35, state="readonly")
        self.combo_doc.pack(side="left", padx=5)
        self.combo_doc.bind("<<ComboboxSelected>>", self.on_doc_selected)
        self.nav_frame.pack_forget()

        row2 = tk.Frame(parent, bg=CLR["surface"])
        row2.pack(fill="x", pady=(8, 0))
        self.info = tk.Label(row2, text="Status: Siap", bg=CLR["surface"], fg=CLR["text_muted"],
                             font=("Segoe UI", 12))
        self.info.pack(side="left")
        self.progress = ttk.Progressbar(row2, length=300, mode="determinate", style="TProgressbar")
        self.progress.pack(side="right")

    # Trace Panel
    def _build_trace_panel(self, parent):
        card = self.create_card(parent, "Trace Langkah-per-Langkah (Per Dokumen)", CLR["purple"])

        # Notebook tabs
        nb_style = ttk.Style()
        nb_style.configure("Step.TNotebook", background=CLR["surface"])
        self.notebook = ttk.Notebook(card, style="Step.TNotebook")
        self.notebook.pack(fill="both", expand=True, pady=(4, 0))

        self.prepro_steps  = {}
        self.step_stat_lbl = {}

        for key, label, expl in STEPS_META:
            tab = tk.Frame(self.notebook, bg=CLR["surface"], padx=0, pady=0)
            self.notebook.add(tab, text=f"  {label}  ")

            # Bug #10 fix: buat label badge dan simpan ke step_stat_lbl
            stat_lbl = tk.Label(
                tab, text="",
                font=("Segoe UI", 8), bg=CLR["surface"],
                fg=CLR["text_muted"], anchor="e"
            )
            stat_lbl.pack(anchor="e", padx=10, pady=(4, 0))
            self.step_stat_lbl[key] = stat_lbl

            # Text widget
            txt = tk.Text(tab, height=7, font=("Consolas", 12), bg=CLR["surface"], fg=CLR["text_body"], relief="flat", 
                          borderwidth=0, wrap="word", padx=10, pady=8, selectbackground=CLR["primary_lt"])
            txt_vsb = ttk.Scrollbar(tab, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=txt_vsb.set)
            txt_vsb.pack(side="right", fill="y")
            txt.pack(fill="both", expand=True)
            self.prepro_steps[key] = txt

        # Bar Chart per Dokumen
        self.trace_bar_frame = tk.Frame(card, bg=CLR["surface"])
        self.trace_bar_frame.pack(fill="x", pady=(0, 12))

        self.trace_fig, self.trace_ax = plt.subplots(figsize=(12, 1.8), dpi=90)
        self.trace_fig.patch.set_facecolor(CLR["surface"])
        self.trace_ax.set_facecolor(CLR["surface2"])
        self._init_trace_bar()
        self.trace_canvas = FigureCanvasTkAgg(self.trace_fig, master=self.trace_bar_frame)
        self.trace_canvas.get_tk_widget().pack(fill="x")

    def _init_trace_bar(self):
        self.trace_ax.clear()
        keys   = [k for k, *_ in STEPS_META]
        labels = [l for _, l, _ in STEPS_META]
        colors = [STEP_COLORS.get(k, CLR["border"]) for k in keys]
        vals   = [0] * len(keys)
        bars = self.trace_ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
        self.trace_ax.set_ylabel("Token", fontsize=7, color=CLR["text_muted"])
        self.trace_ax.set_title("Jumlah Token per Tahap (pilih dokumen)", fontsize=8, color=CLR["text_muted"])
        self.trace_ax.tick_params(labelsize=7)
        for spine in self.trace_ax.spines.values():
            spine.set_edgecolor(CLR["border"])
        self.trace_fig.tight_layout()
        self.trace_canvas.draw() if hasattr(self, "trace_canvas") else None

    def _update_trace_bar(self, token_counts: dict):
        self.trace_ax.clear()
        self.trace_ax.set_facecolor(CLR["surface2"])
        keys   = [k for k, *_ in STEPS_META]
        labels = [l for _, l, _ in STEPS_META]
        colors = [STEP_COLORS.get(k, CLR["border"]) for k in keys]
        vals   = [token_counts.get(k, 0) for k in keys]

        bars = self.trace_ax.bar(labels, vals, color=colors, width=0.6, edgecolor="white", linewidth=0.8, zorder=3)
        self.trace_ax.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)
        for bar, val in zip(bars, vals):
            if val > 0:
                self.trace_ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    str(val), ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=CLR["text_hd"])

        self.trace_ax.set_ylabel("Token", fontsize=7, color=CLR["text_muted"])
        self.trace_ax.set_title("Jumlah Token per Tahap", fontsize=8, color=CLR["text_muted"])
        self.trace_ax.tick_params(labelsize=7)
        for spine in self.trace_ax.spines.values():
            spine.set_edgecolor(CLR["border"])
        self.trace_fig.tight_layout()
        self.trace_canvas.draw()

    # Panel EDA
    def _build_eda_panel(self, parent):
        card = self.create_card(parent, "EDA: Analisis Hasil Pra-pemrosesan", CLR["success"])

        # KPI Row
        self.kpi_row = tk.Frame(card, bg=CLR["surface"])
        self.kpi_row.pack(fill="x", pady=(0, 14))
        self.lbl_grand_total_raw = tk.Label(self.kpi_row, text="Kata Asli: —", bg=CLR["surface"], font=("Segoe UI", 10))
        self.lbl_grand_total_clean = tk.Label(self.kpi_row, text="Kata Bersih: —", bg=CLR["surface"],
                                              font=("Segoe UI", 10, "bold"), fg=CLR["success"])
        self.lbl_efficiency = tk.Label(self.kpi_row, text="Reduksi: —", bg=CLR["surface"], font=("Segoe UI", 10))
        self.lbl_vocab = tk.Label(self.kpi_row, text="Kata Unik: —", bg=CLR["surface"], font=("Segoe UI", 10))
        for lbl in (self.lbl_grand_total_raw, self.lbl_grand_total_clean, self.lbl_efficiency, self.lbl_vocab):
            lbl.pack(side="left", expand=True)

        # Grid 2×2 charts
        charts_area = tk.Frame(card, bg=CLR["surface"])
        charts_area.pack(fill="both", expand=True)
        charts_area.columnconfigure(0, weight=1)
        charts_area.columnconfigure(1, weight=1)

        self.container_a = self._make_chart_card(charts_area, "Reduksi Token per Dokumen", CLR["primary"], 0, 0)
        self.container_b = self._make_chart_card(charts_area, "Distribusi Panjang Dokumen (Bersih)", CLR["purple"], 0, 1)
        self.container_c = self._make_chart_card(charts_area, "15 Kata yang Paling Sering", CLR["teal"], 1, 0)
        self.container_d = self._make_chart_card(charts_area, "Pipeline: Rata-rata Token per Tahap (Aktual)", CLR["orange"], 1, 1)

        self.fig_a, self.ax_a = plt.subplots(figsize=(6, 3.5), dpi=90)
        self.fig_a.patch.set_facecolor(CLR["surface"])
        self.ax_a.set_facecolor(CLR["surface2"])
        self.canvas_a = FigureCanvasTkAgg(self.fig_a, master=self.container_a)
        self.canvas_a.get_tk_widget().pack(fill="both", expand=True)

        self.fig_b, self.ax_b = plt.subplots(figsize=(6, 3.5), dpi=90)
        self.fig_b.patch.set_facecolor(CLR["surface"])
        self.ax_b.set_facecolor(CLR["surface2"])
        self.canvas_b = FigureCanvasTkAgg(self.fig_b, master=self.container_b)
        self.canvas_b.get_tk_widget().pack(fill="both", expand=True)

        self.fig_c, self.ax_c = plt.subplots(figsize=(6, 3.5), dpi=90)
        self.fig_c.patch.set_facecolor(CLR["surface"])
        self.ax_c.set_facecolor(CLR["surface2"])
        self.canvas_c = FigureCanvasTkAgg(self.fig_c, master=self.container_c)
        self.canvas_c.get_tk_widget().pack(fill="both", expand=True)

        self.fig_d, self.ax_d = plt.subplots(figsize=(6, 3.5), dpi=90)
        self.fig_d.patch.set_facecolor(CLR["surface"])
        self.ax_d.set_facecolor(CLR["surface2"])
        self.canvas_d = FigureCanvasTkAgg(self.fig_d, master=self.container_d)
        self.canvas_d.get_tk_widget().pack(fill="both", expand=True)

    def _make_chart_card(self, parent, title, accent_color, row, col):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        card.grid(row=row, column=col, padx=(0 if col == 0 else 6, 0), pady=(0, 10), sticky="nsew")
        hd = tk.Frame(card, bg=CLR["primary_lt"], padx=10, pady=8)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 10, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        content = tk.Frame(card, bg=CLR["surface"], padx=8, pady=8)
        content.pack(fill="both", expand=True)
        return content

    # Logika Pra-pemrosesan
    def run_preprocess(self):
        if self.controller.df is None or self.controller.df.empty:
            return messagebox.showwarning("Peringatan", "Data tidak ditemukan.")
        if "text_raw" not in self.controller.df.columns:
            return messagebox.showwarning("Peringatan", "Kolom text_raw tidak ditemukan. Jalankan OCR terlebih dahulu.")

        n_filled = self.controller.df["text_raw"].notna().sum()
        if n_filled == 0:
            return messagebox.showwarning(
                "Peringatan",
                "Kolom text_raw masih kosong — OCR belum dijalankan.\n"
                "Jalankan Ekstraksi Teks (OCR) terlebih dahulu sebelum Preprocessing."
            )
        self.btn_run.config(state="disabled")
        self.info.config(text="Status: Memproses...", fg=CLR["warning"])
        self.progress.config(value=0)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        df     = self.controller.df
        total  = len(df)
        results            = []
        # untuk grafi rata-rata token
        step_token_sums    = {k: 0 for k, *_ in STEPS_META}
        step_token_sums["Raw"] = 0

        try:
            for i, text in enumerate(df["text_raw"], 1):
                text = str(text)
                cf      = case_folding(text)
                cl      = cleaning(cf)
                tok     = tokenization(cl)
                norm    = normalization(tok)
                sw      = stopword_removal(norm)
                pn      = proper_noun_removal(sw, raw_text=text)
                stem    = stemming(pn)

                # Akumulasi jumlah token per tahap
                step_token_sums["Raw"]  += len(text.split())
                step_token_sums["CF"]   += len(cf.split())
                step_token_sums["Clean"]+= len(cl.split())
                step_token_sums["Tok"]  += len(tok)
                step_token_sums["Norm"] += len(norm)
                step_token_sums["SW"]   += len(sw)
                step_token_sums["PN"]   += len(pn)
                step_token_sums["Stem"] += len(stem)

                results.append(" ".join(stem))
                pct = (i / total) * 100
                self.after(0, lambda v=pct: self.progress.config(value=v))

            # Hitung rata-rata
            step_avg = {k: round(v / total, 1) for k, v in step_token_sums.items()}

            def finish():
                self.controller.df["text_clean"] = results
                self.controller.df["_step_avg"]  = None   # placeholder
                self.controller._step_avg = step_avg
                self.refresh_doc_list()
                self.info.config(text="Status: Selesai!", fg=CLR["success"])
                self.btn_run.config(state="normal")

                # Beritahu SplitFrame agar pool diisi otomatis
                split_frame = getattr(self.controller, "split_frame", None)
                if split_frame is not None and hasattr(split_frame, "on_tab_shown"):
                    split_frame.on_tab_shown()

                messagebox.showinfo("Sukses", f"Preprocessing selesai! {total} dokumen telah diproses.")

            self.after(0, finish)

        except Exception as e:
            err = str(e)
            def on_error():
                messagebox.showerror("Error Preprocessing", f"Proses terhenti:\n{err}")
                self.btn_run.config(state="normal")
                self.info.config(text="Status: Error", fg=CLR["danger"])
            self.after(0, on_error)

    # Refresh dan chart
    def refresh_doc_list(self):
        df = self.controller.df
        if df is None or df.empty or "text_clean" not in df.columns:
            self.nav_frame.pack_forget()
            return

        filenames = df["filename"].tolist() if "filename" in df.columns else \
              [f"Dokumen {i+1}" for i in range(len(df))]
        self.combo_doc["values"] = filenames
        self.nav_frame.pack(side="right", padx=10)
        if filenames and self.combo_doc.get() == "":
            self.combo_doc.current(0)
            self.on_doc_selected(None)

        # Menghitung kata
        df["wc_raw"]   = df["text_raw"].apply(lambda x: len(str(x).split()))
        df["wc_clean"] = df["text_clean"].apply(lambda x: len(str(x).split()))

        all_tokens = []
        for t in df["text_clean"]:
            all_tokens.extend(str(t).split())

        raw_cnt   = int(df["wc_raw"].sum())
        clean_cnt = len(all_tokens)
        reduksi   = ((raw_cnt - clean_cnt) / raw_cnt * 100) if raw_cnt > 0 else 0
        vocab     = len(set(all_tokens))

        self.lbl_grand_total_raw.config(text=f"Kata Asli: {raw_cnt:,}")
        self.lbl_grand_total_clean.config(text=f"Kata Bersih: {clean_cnt:,}")
        self.lbl_efficiency.config( text=f"Reduksi: {reduksi:.1f}%", fg=CLR["success"] if reduksi > 30 else CLR["warning"])
        self.lbl_vocab.config(text=f"🔤 Vocab Unik: {vocab:,}")

        self._render_reduksi_per_dok(df)
        self._render_distribusi_panjang(df)
        self._render_top15_kata(all_tokens)
        step_avg = getattr(self.controller, "_step_avg", None)
        self._render_pipeline_aktual(step_avg)

    # Chart sesudah-sebelum
    def _render_reduksi_per_dok(self, df):
        self.ax_a.clear()
        self.ax_a.set_facecolor(CLR["surface2"])
        raw_counts   = df["wc_raw"].values
        clean_counts = df["wc_clean"].values
        x     = np.arange(len(df))
        width = 0.38
        self.ax_a.bar(x - width / 2, raw_counts,   width, label="Sebelum", color=CLR["warning"], alpha=0.85, zorder=3)
        self.ax_a.bar(x + width / 2, clean_counts, width, label="Sesudah", color=CLR["success"], alpha=0.85, zorder=3)
        self.ax_a.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)
        self.ax_a.set_xlabel("Dokumen ke-", fontsize=8, color=CLR["text_muted"])
        self.ax_a.set_ylabel("Jumlah Token",  fontsize=8, color=CLR["text_muted"])
        self.ax_a.set_title("Sebelum vs Sesudah Preprocessing per Dokumen", fontsize=9, fontweight="bold", color=CLR["text_hd"])
        self.ax_a.legend(fontsize=8, framealpha=0.8)
        self.ax_a.tick_params(labelsize=7)
        self.ax_a.spines["top"].set_visible(False)
        self.ax_a.spines["right"].set_visible(False)
        self.fig_a.tight_layout()
        self.canvas_a.draw()

    # Chart distribusi panjang dokumen
    def _render_distribusi_panjang(self, df):
        self.ax_b.clear()
        self.ax_b.set_facecolor(CLR["surface2"])
        vals    = df["wc_clean"].values
        bins    = min(20, max(1, len(np.unique(vals))))
        self.ax_b.hist(vals, bins=bins, color=CLR["purple"], edgecolor="white", alpha=0.85, zorder=3)
        mean_v   = float(np.mean(vals))
        median_v = float(np.median(vals))
        self.ax_b.axvline(mean_v,   color=CLR["danger"],  linestyle="--", linewidth=2, label=f"Mean: {mean_v:.0f}")
        self.ax_b.axvline(median_v, color=CLR["primary"], linestyle="-.", linewidth=2, label=f"Median: {median_v:.0f}")
        self.ax_b.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)
        self.ax_b.set_xlabel("Jumlah Token", fontsize=8, color=CLR["text_muted"])
        self.ax_b.set_ylabel("Frekuensi",    fontsize=8, color=CLR["text_muted"])
        self.ax_b.set_title("Distribusi Panjang Dokumen", fontsize=9, fontweight="bold", color=CLR["text_hd"])
        self.ax_b.legend(fontsize=8)
        self.ax_b.tick_params(labelsize=7)
        self.ax_b.spines["top"].set_visible(False)
        self.ax_b.spines["right"].set_visible(False)
        self.fig_b.tight_layout()
        self.canvas_b.draw()

    # Chart kata yang sering muncul
    def _render_top15_kata(self, tokens):
        self.ax_c.clear()
        self.ax_c.set_facecolor(CLR["surface2"])
        most_common = Counter(tokens).most_common(15)
        if most_common:
            words, counts = zip(*most_common)
            import matplotlib.cm as cm
            norm_c = [c / max(counts) for c in counts]
            bar_colors = [
                "#{:02X}{:02X}{:02X}".format(
                    int(20  + (20 - 20) * (1 - n)),
                    int(184 + (100 - 184) * (1 - n)),
                    int(166 + (230 - 166) * (1 - n))
                )
                for n in norm_c
            ]
            bars = self.ax_c.barh(words, counts, color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)
            self.ax_c.invert_yaxis()
            for bar, val in zip(bars, counts):
                self.ax_c.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                               str(val), va="center", fontsize=8, color=CLR["text_body"], fontweight="bold")
        self.ax_c.grid(axis="x", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)
        self.ax_c.set_xlabel("Frekuensi", fontsize=8, color=CLR["text_muted"])
        self.ax_c.set_title("15 Kata Paling Sering (Setelah Stemming)", fontsize=9, fontweight="bold", color=CLR["text_hd"])
        self.ax_c.tick_params(labelsize=8)
        self.ax_c.spines["top"].set_visible(False)
        self.ax_c.spines["right"].set_visible(False)
        self.fig_c.tight_layout()
        self.canvas_c.draw()

    # Chart rata-rata token
    def _render_pipeline_aktual(self, step_avg: dict):
        self.ax_d.clear()
        self.ax_d.set_facecolor(CLR["surface2"])

        if not step_avg:
            self.ax_d.text(0.5, 0.5, "Jalankan preprocessing terlebih dahulu", ha="center", va="center", 
                           transform=self.ax_d.transAxes, fontsize=9, color=CLR["text_muted"])
            self.fig_d.tight_layout()
            self.canvas_d.draw()
            return

        keys   = [k for k, *_ in STEPS_META]
        labels = [l for _, l, _ in STEPS_META]
        colors = [STEP_COLORS.get(k, CLR["border"]) for k in keys]
        vals   = [step_avg.get(k, 0) for k in keys]

        bars = self.ax_d.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8, zorder=3)
        self.ax_d.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)

        for bar, val in zip(bars, vals):
            if val > 0:
                self.ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01, f"{val:.0f}", 
                               ha="center", va="bottom", fontsize=7, fontweight="bold", color=CLR["text_hd"])

        self.ax_d.set_ylabel("Rata-rata Token", fontsize=8, color=CLR["text_muted"])
        self.ax_d.set_title("Penurunan Rata-rata Token per Tahap (Aktual)", fontsize=9, fontweight="bold", color=CLR["text_hd"])
        self.ax_d.tick_params(axis="x", rotation=20, labelsize=7)
        self.ax_d.tick_params(axis="y", labelsize=7)
        self.ax_d.spines["top"].set_visible(False)
        self.ax_d.spines["right"].set_visible(False)
        self.fig_d.tight_layout()
        self.canvas_d.draw()

    # Trace untuk dokumen yang dipilih
    def on_doc_selected(self, event):
        idx = self.combo_doc.current()
        if idx == -1:
            return
        row = self.controller.df.iloc[idx]
        raw = str(row["text_raw"])
        cf   = case_folding(raw)
        cl   = cleaning(cf)
        tok  = tokenization(cl)
        norm = normalization(tok)
        sw   = stopword_removal(norm)
        pn   = proper_noun_removal(sw, raw_text=raw)
        stem = stemming(pn)

        steps_content = {
            "Raw":  (raw,              len(raw.split())),
            "CF":   (cf,               len(cf.split())),
            "Clean":(cl,               len(cl.split())),
            "Tok":  (", ".join(tok),   len(tok)),
            "Norm": (" ".join(norm),   len(norm)),
            "SW":   (" ".join(sw),     len(sw)),
            "PN":   (" ".join(pn),     len(pn)),
            "Stem": (" ".join(stem),   len(stem)),
        }

        prev_count = None
        for key, (content, count) in steps_content.items():
            self._set_text(key, content)
            # Badge: token count + delta dari tahap sebelumnya
            if prev_count is None:
                badge = f"{count:,} token"
            else:
                delta = count - prev_count
                sign  = "+" if delta >= 0 else ""
                delta_str = f"  ({sign}{delta})"
                badge = f"{count:,} token{delta_str}"
            lbl = self.step_stat_lbl.get(key)
            if lbl:
                lbl.config(text=badge,fg=CLR["success"] 
                           if (prev_count and delta < 0)
                           else STEP_COLORS.get(key, CLR["primary"]))
            prev_count = count

        # Update mini bar chart
        self._update_trace_bar({k: v[1] for k, v in steps_content.items()})

        # Navigasi ke tab Raw
        self.notebook.select(0)

    def _set_text(self, key, text):
        widget = self.prepro_steps.get(key)
        if widget is None:
            return
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)