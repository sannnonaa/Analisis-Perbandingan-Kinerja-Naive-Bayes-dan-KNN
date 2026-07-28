import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
import pandas as pd
from feature_extraction import fit_transform_tfidf, get_tfidf_detail
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

CLR = {
    "bg":           "#F0F2F8", "surface":      "#FFFFFF", "surface2":     "#F8FAFF", "primary":      "#4F46E5", "primary_dk":   "#3730A3", 
    "primary_lt":   "#EEF2FF", "success":      "#10B981", "warning":      "#F59E0B", "danger":       "#EF4444", "purple":       "#8B5CF6", 
    "blue":         "#3B82F6", "text_hd":      "#1E1B4B", "text_body":    "#374151", "text_muted":   "#6B7280", "border":       "#E5E7EB",
    "teal":         "#14B8A6", "orange":       "#F97316",
}

CLASS_COLORS = {
    "Sertifikat":  "#4F46E5", "Persetujuan": "#10B981", "Perbaikan":   "#F59E0B", "Perizinan":   "#8B5CF6",
}
DEFAULT_CLASS_COLORS = ["#4F46E5", "#10B981", "#F59E0B", "#8B5CF6","#EF4444", "#3B82F6", "#F97316", "#14B8A6"]


class TfidfFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR["bg"])
        self.controller     = controller
        self.X_train_tfidf  = None
        self.X_test_tfidf   = None
        self.feature_names  = None
        self._tfidf_cache   = {}
        self._tfidf_cache_test = {}
        self._train_indices = []
        self._test_indices  = []
        self._per_class_canvas = None  # Bug #5 fix: referensi canvas grafik per-kelas

        self._apply_styles()
        self._build_header()

        container = tk.Frame(self, bg=CLR["bg"])
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, bg=CLR["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=CLR["bg"], padx=25, pady=20)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        # 1. Panel kontrol
        self.card_control = self._create_control_card(self.scroll_frame, CLR["primary"])
        self._setup_control_ui(self.card_control)

        # 2. KPI
        self.kpi_container = tk.Frame(self.scroll_frame, bg=CLR["bg"])
        self.kpi_container.pack(fill="x", pady=(0, 20))

        # 5. Tabel rincian per dokumen — DATA LATIH
        self.card_main = self._create_card(self.scroll_frame, "Rincian Kalkulasi TF-IDF per Dokumen — Data Latih", CLR["blue"], expand=False)
        tables_row = tk.Frame(self.card_main, bg=CLR["surface"])
        tables_row.pack(fill="both", expand=True)
        tables_row.columnconfigure(0, weight=1)
        tables_row.columnconfigure(1, weight=3)

        left_panel = tk.Frame(tables_row, bg=CLR["surface"])
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(left_panel, text="Kata Dasar (Token Unik)", bg=CLR["primary_lt"], fg=CLR["purple"], font=("Segoe UI", 9, "bold"), anchor="w",
                 padx=10, pady=6).pack(fill="x")
        self.table_stem = self._build_stem_table(left_panel)

        right_panel = tk.Frame(tables_row, bg=CLR["surface"])
        right_panel.grid(row=0, column=1, sticky="nsew")
        tk.Label(right_panel, text="TF-IDF per Term  (TF × IDF = Skor)  —  IDF dihitung dari data latih",
                 bg=CLR["primary_lt"], fg=CLR["blue"], font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        self.table_main = self._build_table(right_panel, [("term", "Term / N-gram", 200), ("tf", "TF (raw)", 70),
                                                          ("tf_log", "TF (log)", 80), ("df", "DF", 60),
                                                          ("idf", "IDF", 100), ("score", "TF-IDF", 110)])

        # 6. Tabel rincian per dokumen — DATA UJI
        self.card_test = self._create_card(self.scroll_frame, "Rincian Kalkulasi TF-IDF per Dokumen — Data Uji", CLR["teal"], expand=False)

        # Keterangan data leakage
        note_frame = tk.Frame(self.card_test, bg="#E6F7F4", padx=12, pady=8,
                              highlightbackground=CLR["teal"], highlightthickness=1)
        note_frame.pack(fill="x", pady=(0, 10))
        tk.Label(note_frame,
                 text="ℹ  IDF berasal dari data latih (tidak dihitung ulang). Hanya TF yang dihitung dari dok uji. Mencegah data leakage.",
                 bg="#E6F7F4", fg=CLR["teal"], font=("Segoe UI", 8), justify="left").pack(anchor="w")

        tables_row_test = tk.Frame(self.card_test, bg=CLR["surface"])
        tables_row_test.pack(fill="both", expand=True)
        tables_row_test.columnconfigure(0, weight=1)
        tables_row_test.columnconfigure(1, weight=3)

        left_panel_test = tk.Frame(tables_row_test, bg=CLR["surface"])
        left_panel_test.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(left_panel_test, text="Kata Dasar (Token Unik)", bg=CLR["primary_lt"], fg=CLR["purple"],
                 font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        self.table_stem_test = self._build_stem_table(left_panel_test)

        right_panel_test = tk.Frame(tables_row_test, bg=CLR["surface"])
        right_panel_test.grid(row=0, column=1, sticky="nsew")
        tk.Label(right_panel_test,
                 text="TF-IDF per Term  —  IDF dari data latih, TF dari dokumen uji",
                 bg=CLR["primary_lt"], fg=CLR["teal"], font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        self.table_test = self._build_table(right_panel_test, [("term", "Term / N-gram", 200), ("tf", "TF (raw)", 70),
                                                               ("tf_log", "TF (log)", 80), ("df", "DF (latih)", 60),
                                                               ("idf", "IDF (latih)", 100), ("score", "TF-IDF", 110)])

        # Selector dokumen uji
        self.nav_frame_test = tk.Frame(self.card_test, bg=CLR["surface2"], padx=15, pady=6,
                                       highlightbackground=CLR["border"], highlightthickness=1)
        tk.Label(self.nav_frame_test, text="Trace Dokumen (Test):", bg=CLR["surface2"],
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        self.combo_doc_test = ttk.Combobox(self.nav_frame_test, width=35, state="readonly", font=("Segoe UI", 9))
        self.combo_doc_test.pack(side="left", padx=5)
        self.combo_doc_test.bind("<<ComboboxSelected>>", self.on_test_doc_selected)
        self.nav_frame_test.pack_forget()

    # Styles
    def _apply_styles(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TProgressbar", thickness=10, troughcolor=CLR["bg"], background=CLR["primary"], borderwidth=0)
        s.configure("Modern.Treeview", background=CLR["surface"], fieldbackground=CLR["surface"], rowheight=24, font=("Segoe UI", 9))
        s.configure("Modern.Treeview.Heading", background=CLR["primary_lt"], foreground=CLR["primary"], font=("Segoe UI", 9, "bold"))
        s.map("Modern.Treeview", background=[("selected", CLR["primary_lt"])], foreground=[("selected", CLR["primary"])])

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # Header
    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(side="top", fill="x")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(inner, text="Tahap 5: Ekstraksi Fitur & Pembobotan TF-IDF", font=("Segoe UI", 18, "bold"), bg=CLR["surface"], fg=CLR["text_hd"]
                 ).pack(side="left", anchor="w")
        tk.Label(inner, text="ngram (1,2)  ·  min_df adaptif  ·  max_df=0.70  ·  sublinear_tf=True", font=("Segoe UI", 9),
                 bg=CLR["surface"], fg=CLR["text_muted"]).pack(side="left", padx=14, anchor="w")

    # Panel Controll
    def _setup_control_ui(self, parent):
        top_row = tk.Frame(parent, bg=CLR["surface"])
        top_row.pack(fill="x", expand=True)

        self.btn_run = tk.Button(top_row, text="Bangun Fitur TF-IDF", command=self.run_tfidf, bg=CLR["primary"], fg="white",
                                 font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", padx=20, pady=10)
        self.btn_run.pack(side="left")

        param_frame = tk.Frame(top_row, bg=CLR["surface"], padx=15)
        param_frame.pack(side="left")

        tk.Label(param_frame, text="Min DF:", bg=CLR["surface"], font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        self.min_df_slider = tk.Scale(param_frame, from_=1, to=10, orient="horizontal", bg=CLR["surface"], length=90, activebackground=CLR["primary"])
        self.min_df_slider.set(2)
        self.min_df_slider.grid(row=0, column=1, padx=(0, 6))

        self.min_df_auto_var = tk.BooleanVar(value=False)
        self.min_df_auto_cb = tk.Checkbutton(param_frame, text="Auto min_df", variable=self.min_df_auto_var, bg=CLR["surface"],
                                            command=lambda: self.min_df_slider.config(state=("disabled" if self.min_df_auto_var.get() else "normal")))
        self.min_df_auto_cb.grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 6))

        tk.Label(param_frame, text="Max DF (%):", bg=CLR["surface"], font=("Segoe UI", 9)).grid(row=0, column=2, padx=(10, 0), sticky="w")
        self.max_df_slider = tk.Scale(param_frame, from_=50, to=100, orient="horizontal", bg=CLR["surface"], length=90, activebackground=CLR["primary"])
        self.max_df_slider.set(70)
        self.max_df_slider.grid(row=0, column=3, padx=(0, 6))

        tk.Label(param_frame, text="ngram (1,2)  ·  sublinear_tf=True  [tetap]", bg=CLR["surface"], font=("Segoe UI", 8),
                  fg=CLR["text_muted"]).grid(row=0, column=4, padx=(14, 0), sticky="w")

        self.nav_frame = tk.Frame(top_row, bg=CLR["surface2"], padx=15, pady=6, highlightbackground=CLR["border"], highlightthickness=1)
        tk.Label(self.nav_frame, text="Trace Dokumen (Train):", bg=CLR["surface2"], font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 10))
        self.combo_doc = ttk.Combobox(self.nav_frame, width=35, state="readonly", font=("Segoe UI", 9))
        self.combo_doc.pack(side="left", padx=5)
        self.combo_doc.bind("<<ComboboxSelected>>", self.on_doc_selected)
        self.nav_frame.pack_forget()

        bottom_row = tk.Frame(parent, bg=CLR["surface"])
        bottom_row.pack(fill="x", pady=(12, 0))
        self.info = tk.Label(bottom_row, text="Status: Siap", bg=CLR["surface"], fg=CLR["text_muted"], font=("Segoe UI", 9, "bold"))
        self.info.pack(side="left")
        self.progress = ttk.Progressbar(bottom_row, length=250, mode="indeterminate", style="TProgressbar")
        self.progress.pack(side="right")

    # Card
    def _create_control_card(self, parent, accent_color):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        card.pack(fill="x", pady=(0, 20))
        tk.Frame(card, bg=accent_color, height=4).pack(fill="x")
        content = tk.Frame(card, bg=CLR["surface"], padx=20, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _create_card(self, parent, title, accent_color, expand=False):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="x", expand=False, pady=(0, 20))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _build_stem_table(self, parent):
        container = tk.Frame(parent, bg=CLR["surface"])
        container.pack(fill="both", expand=True)
        tree = ttk.Treeview(container, columns=("no", "kata"), show="headings", height=12, style="Modern.Treeview")
        tree.heading("no",   text="#")
        tree.heading("kata", text="Token")
        tree.column("no",   width=40,  anchor="center")
        tree.column("kata", width=160)
        sb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    def _build_table(self, parent, columns):
        container = tk.Frame(parent, bg=CLR["surface"])
        container.pack(fill="both", expand=True)
        cols = [c[0] for c in columns]
        tree = ttk.Treeview(container, columns=cols, show="headings", height=12, style="Modern.Treeview")
        for cid, label, width in columns:
            tree.heading(cid, text=label)
            tree.column(cid, width=width, anchor="w" if "term" in cid else "center")
        sb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree



    # KPI
    def _render_kpis(self, total_fitur, total_latih, total_uji, total_words, rejected, raw_count, ngram, min_df_used):
        for w in self.kpi_container.winfo_children():
            w.destroy()
        kpis = [
            ("Total Fitur\n(Fitur Aktif)", f"{total_fitur:,}",
             CLR["primary"], CLR["primary_lt"]),
            ("Fitur yang Dibuang", f"{rejected:,}",
             CLR["danger"], "#FEF2F2"),
            ("N-gram Range", f"{ngram}",
             CLR["blue"], "#EFF6FF"),
            ("Train : Test", f"{total_latih} : {total_uji}",
             CLR["success"], "#F0FDF4"),
        ]
        for i, (title, value, fg, bg) in enumerate(kpis):
            kf = tk.Frame(self.kpi_container, bg=bg, highlightbackground=fg, highlightthickness=2)
            kf.pack(side="left", fill="both", expand=True, padx=(0, 8 if i < len(kpis) - 1 else 0))
            tk.Frame(kf, bg=fg, height=4).pack(fill="x")
            tk.Label(kf, text=value, font=("Segoe UI", 15, "bold"), bg=bg, fg=fg).pack(pady=(10, 2))
            tk.Label(kf, text=title, font=("Segoe UI", 8), bg=bg, fg=fg, justify="center", wraplength=110).pack(pady=(0, 10))

    def run_tfidf(self):
        if not hasattr(self.controller, "train_idx") or self.controller.train_idx is None:
            messagebox.showwarning("Peringatan", "Lakukan pembagian data di Tahap 4 terlebih dahulu.")
            return
        if self.controller.df is None or "text_clean" not in self.controller.df.columns:
            messagebox.showwarning("Peringatan", "Lakukan preprocessing terlebih dahulu.")
            return

        self._tfidf_cache = {}
        self.progress.start(10)
        self.btn_run.config(state="disabled")
        self.info.config(text="Status: Memproses...", fg=CLR["warning"])

        min_df = None if getattr(self, "min_df_auto_var", None) and self.min_df_auto_var.get() else self.min_df_slider.get()
        max_df = self.max_df_slider.get() / 100.0

        def worker():
            try:
                df          = self.controller.df
                tr_idx      = self.controller.train_idx
                ts_idx      = self.controller.test_idx
                texts       = df["text_clean"].fillna("").tolist()
                train_texts = [texts[i] for i in tr_idx]
                test_texts  = [texts[i] for i in ts_idx]

                # Raw vocab count (min_df=1, max_df=1.0) untuk hitung rejected
                _, vec_raw  = fit_transform_tfidf(train_texts, ngram_range=(1, 2), min_df=1, max_df=1.0)
                raw_count   = len(vec_raw.get_feature_names_out())

                # fit_transform HANYA pada TRAIN (prevent leakage)
                X_train, vec = fit_transform_tfidf(train_texts, ngram_range=(1, 2), min_df=min_df, max_df=max_df)

                # transform TEST dengan vocab train
                X_test        = vec.transform(test_texts)
                feature_names = vec.get_feature_names_out()
                rejected      = raw_count - len(feature_names)
                total_words   = sum(len(t.split()) for t in train_texts)
                # Gunakan nilai min_df yang dikirim ke vectorizer, bukan yang tersimpan di sklearn
                # (vec.min_df bisa berbeda jika adaptif, sehingga tidak mencerminkan pilihan user)
                min_df_used   = min_df if min_df is not None else getattr(vec, "min_df", min_df)

                def finish():
                    self.X_train_tfidf  = X_train
                    self.X_test_tfidf   = X_test
                    self.feature_names  = feature_names
                    self._train_indices = list(tr_idx)

                    self.controller.X_train    = X_train
                    self.controller.X_test     = X_test
                    self.controller.vectorizer = vec

                    # Populate real-only subsets (needed for valid synthetic impact analysis)
                    if "source_type" in df.columns:
                        src = df["source_type"].fillna("unknown").values
                    else:
                        src = np.array(["unknown"] * len(df))
                    labels = df["label"].values
                    train_src = src[tr_idx]
                    test_src = src[ts_idx]

                    real_train_mask = (train_src == "real")
                    real_test_mask = (test_src == "real")

                    if real_train_mask.any():
                        self.controller.X_real_train = X_train[real_train_mask]
                        self.controller.y_real_train = labels[tr_idx][real_train_mask]
                        # Backward-compatible single-name attributes
                        self.controller.X_real = self.controller.X_real_train
                        self.controller.y_real = self.controller.y_real_train
                    else:
                        self.controller.X_real_train = None
                        self.controller.y_real_train = None
                        self.controller.X_real = None
                        self.controller.y_real = None

                    if real_test_mask.any():
                        self.controller.X_real_test = X_test[real_test_mask]
                        self.controller.y_real_test = labels[ts_idx][real_test_mask]
                    else:
                        self.controller.X_real_test = None
                        self.controller.y_real_test = None

                    # Also keep full train copy for convenience
                    self.controller.X_full_train = X_train
                    self.controller.y_full_train = labels[tr_idx]


                    # Save raw text and label lists for text-based analysis
                    self.controller.full_train_texts  = train_texts
                    self.controller.full_train_labels = labels[tr_idx]
                    self.controller.full_test_texts   = test_texts
                    self.controller.full_test_labels  = labels[ts_idx]

                    if real_train_mask.any():
                        real_train_indices = np.where(real_train_mask)[0]
                        self.controller.real_train_texts = [train_texts[i] for i in real_train_indices]
                        self.controller.real_train_labels = self.controller.y_real_train
                    else:
                        self.controller.real_train_texts = []
                        self.controller.real_train_labels = None

                    if real_test_mask.any():
                        real_test_indices = np.where(real_test_mask)[0]
                        self.controller.real_test_texts = [test_texts[i] for i in real_test_indices]
                        self.controller.real_test_labels = self.controller.y_real_test
                    else:
                        self.controller.real_test_texts = []
                        self.controller.real_test_labels = None

                    # Isi selector dokumen latih
                    self.nav_frame.pack(side="right")
                    self.combo_doc["values"] = df.iloc[tr_idx]["filename"].tolist()
                    self.combo_doc.current(0)
                    self.on_doc_selected(None)

                    # Isi selector dokumen uji
                    self._test_indices = list(ts_idx)
                    self._tfidf_cache_test = {}
                    self.nav_frame_test.pack(fill="x", pady=(8, 0))
                    self.combo_doc_test["values"] = df.iloc[ts_idx]["filename"].tolist()
                    self.combo_doc_test.current(0)
                    self.on_test_doc_selected(None)

                    # KPI dengan parameter aktual
                    self._render_kpis(
                        len(feature_names), len(tr_idx), len(ts_idx),
                        total_words, rejected, raw_count,
                        "(1, 2)", min_df_used)


                    # Refresh scrollregion
                    self.scroll_frame.update_idletasks()
                    self.canvas.configure(scrollregion=self.canvas.bbox("all"))

                    self.progress.stop()
                    self.btn_run.config(state="normal")
                    self.info.config(text="Status: ✅ Selesai", fg=CLR["success"])
                    messagebox.showinfo("Sukses", "Ekstraksi fitur selesai.")

                self.after(0, finish)

            except Exception as e:
                msg = str(e)
                def on_error():
                    messagebox.showerror("Error", msg)
                    self.btn_run.config(state="normal")
                    self.progress.stop()
                    self.info.config(text="Status: Error", fg=CLR["danger"])
                self.after(0, on_error)

        threading.Thread(target=worker, daemon=True).start()

    # Tabel trace dokumen
    def on_doc_selected(self, event):
        combo_idx = self.combo_doc.current()
        if combo_idx == -1 or self.X_train_tfidf is None:
            return

        if combo_idx in self._tfidf_cache:
            sorted_data, tokens = self._tfidf_cache[combo_idx]
        else:
            row_vec      = self.X_train_tfidf[combo_idx].toarray()[0]
            nz_indices   = np.where(row_vec > 0)[0]
            df_idx       = self._train_indices[combo_idx]
            clean_text   = str(self.controller.df.iloc[df_idx]["text_clean"])
            idf_vals     = getattr(self.controller.vectorizer, "idf_", None)
            words_in_doc = clean_text.split()

            # Document Frequency: jumlah dokumen yang mengandung tiap term
            # Dihitung dari: IDF = log((1+N)/(1+df)) + 1  →  df = (1+N)/exp(idf-1) - 1
            n_docs = self.X_train_tfidf.shape[0]

            calc_list = []
            for f_idx in nz_indices:
                term       = self.feature_names[f_idx]
                term_words = term.split()
                if len(term_words) == 1:
                    tf = words_in_doc.count(term)
                else:
                    tf = sum(
                        1 for i in range(len(words_in_doc) - len(term_words) + 1)
                        if words_in_doc[i:i + len(term_words)] == term_words)
                idf_val = idf_vals[f_idx] if idf_vals is not None else 0
                # Hitung balik DF dari nilai IDF (smooth IDF sklearn)
                df_val = round((1 + n_docs) / np.exp(idf_val - 1) - 1) if idf_val > 0 else 0
                calc_list.append({
                    "t": term, "tf": tf,
                    "tf_log": round(1 + np.log(tf), 4) if tf > 0 else 0,
                    "df": int(df_val),
                    "idf": idf_val,
                    "score": row_vec[f_idx],
                })

            sorted_data = sorted(calc_list, key=lambda x: x["t"])
            tokens      = sorted(set(words_in_doc))
            self._tfidf_cache[combo_idx] = (sorted_data, tokens)

        for item in self.table_stem.get_children():
            self.table_stem.delete(item)
        for i, tok in enumerate(tokens, 1):
            self.table_stem.insert("", "end", values=(i, tok))

        for item in self.table_main.get_children():
            self.table_main.delete(item)
        for d in sorted_data:
            self.table_main.insert(
                "", "end",
                values=(d["t"], d["tf"], f"{d['tf_log']:.4f}",
                        d["df"], f"{d['idf']:.4f}", f"{d['score']:.4f}"))

    # Tabel trace dokumen uji
    def on_test_doc_selected(self, event):
        combo_idx = self.combo_doc_test.current()
        if combo_idx == -1 or self.X_test_tfidf is None:
            return

        if combo_idx in self._tfidf_cache_test:
            sorted_data, tokens = self._tfidf_cache_test[combo_idx]
        else:
            df        = self.controller.df
            ts_idx    = self._test_indices
            df_idx    = ts_idx[combo_idx]
            clean_text = str(df.iloc[df_idx]["text_clean"])

            # Gunakan get_tfidf_detail — IDF tetap dari vectorizer latih
            detail    = get_tfidf_detail(clean_text, self.controller.vectorizer)
            sorted_data = detail
            tokens    = sorted(set(clean_text.split()))
            self._tfidf_cache_test[combo_idx] = (sorted_data, tokens)

        for item in self.table_stem_test.get_children():
            self.table_stem_test.delete(item)
        for i, tok in enumerate(tokens, 1):
            self.table_stem_test.insert("", "end", values=(i, tok))

        for item in self.table_test.get_children():
            self.table_test.delete(item)
        for d in sorted_data:
            self.table_test.insert(
                "", "end",
                values=(d["term"], d["tf_raw"], f"{d['tf_log']:.4f}",
                        d["df"], f"{d['idf']:.4f}", f"{d['tfidf']:.4f}"))

