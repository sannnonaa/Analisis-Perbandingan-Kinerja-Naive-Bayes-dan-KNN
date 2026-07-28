import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import StratifiedKFold
from evaluation import accuracy_score_manual, classification_report_manual, weighted_f1

CLR = {
    "bg":          "#F0F2F8", "surface":     "#FFFFFF", "surface2":    "#F8FAFF", "primary":     "#4F46E5", "primary_dk":  "#3730A3",
    "primary_lt":  "#EEF2FF", "success":     "#10B981", "warning":     "#F59E0B", "danger":      "#EF4444", "purple":      "#8B5CF6",
    "blue":        "#3B82F6", "text_hd":     "#1E1B4B", "text_body":   "#374151", "text_muted":  "#6B7280", "border":      "#E5E7EB",
    "orange":      "#F97316", "teal":        "#14B8A6",
}


class AnalysisFrame(tk.Frame):
    def __init__(self, parent, controller, modeling_frame=None):
        super().__init__(parent, bg=CLR["bg"])
        self.controller      = controller
        self.modeling_frame  = modeling_frame

        # Runtime state
        self._y_test      = None
        self._nb_pred     = None
        self._knn_pred    = None
        self._last_cv_res = None   # hasil K-Fold terakhir

        self._build_header()

        # Canvas / Scrollbar
        self.canvas    = tk.Canvas(self, bg=CLR["bg"], highlightthickness=0)
        scrollbar      = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=CLR["bg"], padx=28, pady=22)

        self._canvas_win = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._canvas_win, width=e.width)
        )
        self.bind("<Map>",   lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.bind("<Unmap>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        sf = self.scroll_frame
        sf.columnconfigure(0, weight=1, uniform="col")
        sf.columnconfigure(1, weight=1, uniform="col")

        # ROW 0: Confusion Matrix
        nb_cm_card = self._card(sf, "Confusion Matrix — Naive Bayes", 0, 0, accent=CLR["purple"])
        # BUG FIX #2: frame tinggi tetap + pack_propagate False sudah benar,
        # tapi widget matplotlib harus fill="x" bukan fill="both" agar tidak melar
        self.nb_matrix_frame = tk.Frame(nb_cm_card, bg=CLR["surface"], height=380)
        self.nb_matrix_frame.pack(fill="both", expand=True)
        self.nb_matrix_frame.pack_propagate(False)

        knn_cm_card = self._card(sf, "Confusion Matrix — KNN", 0, 1, accent=CLR["blue"])
        self.knn_matrix_frame = tk.Frame(knn_cm_card, bg=CLR["surface"], height=380)
        self.knn_matrix_frame.pack(fill="both", expand=True)
        self.knn_matrix_frame.pack_propagate(False)

        metric_card = self._card(sf, "Metrik Evaluasi Lengkap (Macro F1 sebagai metrik utama)", 1, 0, colspan=2, accent=CLR["primary"])
        self._setup_full_metrics_ui(metric_card)

        error_card = self._card(sf, "Detail Kesalahan Prediksi Dokumen", 2, 0, colspan=2, accent=CLR["warning"])
        self._build_error_table(error_card)

        # ROW 3: K-Fold Control
        kfold_ctrl_card = self._card(sf, "Evaluasi Stratified K-Fold Cross Validation", 3, 0, colspan=2, accent=CLR["success"])
        self._setup_kfold_control_ui(kfold_ctrl_card)

        # ROW 4: K-Fold Stats Table
        stat_card = self._card(sf, "Tabel Statistik Per Fold", 4, 0, colspan=2, accent=CLR["purple"])
        self._setup_kfold_stats_ui(stat_card)

        # ROW 5: K-Fold Trend Chart
        chart_card = self._card(sf, "Grafik Tren Akurasi & Macro F1 Per Fold", 5, 0, colspan=2, accent=CLR["danger"])
        self.kfold_chart_frame = tk.Frame(chart_card, bg=CLR["surface"], height=320)
        self.kfold_chart_frame.pack(fill="both", expand=True)
        self.kfold_chart_frame.pack_propagate(False)

        # ROW 6: Confusion Matrix Per Fold
        cm_fold_card = self._card(sf, "Confusion Matrix Per Fold", 6, 0, colspan=2, accent=CLR["purple"])
        self._setup_kfold_cm_ui(cm_fold_card)

        # ROW 7: Holdout vs K-Fold Comparison
        holdout_cv_card = self._card(sf, "Perbandingan Holdout Test vs K-Fold CV", 7, 0, colspan=2, accent=CLR["orange"])
        self._setup_holdout_cv_ui(holdout_cv_card)

    # ─────────────────────────────────────────────────────────────────
    # HELPER WIDGETS
    # ─────────────────────────────────────────────────────────────────
    def _card(self, parent, title, row, col, colspan=1, accent=CLR["primary"]):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=10, pady=(0, 20))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=18, pady=12)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent).pack(side="left")
        tk.Frame(outer, bg=accent, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=20)
        content.pack(fill="both", expand=True)
        return content

    def _section_label(self, parent, text, color):
        f = tk.Frame(parent, bg=CLR["surface"])
        f.pack(fill="x", pady=(14, 4))
        tk.Frame(f, bg=color, width=4).pack(side="left", fill="y")
        tk.Label(f, text=f"  {text}", font=("Segoe UI", 10, "bold"), fg=color, bg=CLR["surface"]).pack(side="left", anchor="w")

    def _info_box(self, parent, text, color=CLR["primary"]):
        bg = CLR["primary_lt"]
        box = tk.Frame(parent, bg=bg, padx=12, pady=8, highlightbackground=color, highlightthickness=1)
        box.pack(fill="x", pady=(0, 10))
        tk.Label(box, text=text, font=("Courier New", 11), fg=CLR["text_body"], bg=bg, justify="left", wraplength=900).pack(anchor="w")

    def _build_header(self):
        header = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        header.pack(side="top", fill="x")
        tk.Label(
            header, text="Tahap 7: Analisis Kinerja Model",
            font=("Segoe UI", 18, "bold"),
            bg=CLR["surface"], fg=CLR["text_hd"],
            padx=20, pady=14
        ).pack(side="left")

    def _draw_matrix(self, y_t, y_p, parent, cmap, model_name):
        for w in parent.winfo_children():
            w.destroy()

        cls = sorted(set(y_t) | set(y_p))
        mat = np.zeros((len(cls), len(cls)), dtype=int)
        idx = {c: i for i, c in enumerate(cls)}
        for t, p in zip(y_t, y_p):
            mat[idx[t]][idx[p]] += 1

        fig = Figure(figsize=(4.4, 3.6), dpi=95)
        ax  = fig.add_subplot(111)
        ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(cls)))
        ax.set_yticklabels(cls, fontsize=8)
        ax.set_xlabel("Prediksi", fontsize=9)
        ax.set_ylabel("Aktual",   fontsize=9)
        ax.set_title(model_name, fontsize=10, weight="bold")

        max_val = mat.max() if mat.max() > 0 else 1
        for i in range(len(cls)):
            for j in range(len(cls)):
                val = int(mat[i, j])
                color = "white" if val > max_val * 0.5 else "black"
                ax.text(j, i, val, ha="center", va="center", weight="bold", fontsize=9, color=color)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

    def _setup_full_metrics_ui(self, card):
        self.full_metrics_frame = card
        tk.Label(card,
                 text="Latih dan uji model terlebih dahulu untuk melihat metrik lengkap.",
                 font=("Segoe UI", 9, "italic"),
                 fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=10)

    def _render_full_metrics(self, y_test, nb_pred, knn_pred):
        for w in self.full_metrics_frame.winfo_children():
            w.destroy()

        nb_rep,  _, _ = classification_report_manual(y_test, nb_pred)
        knn_rep, _, _ = classification_report_manual(y_test, knn_pred)
        nb_acc    = accuracy_score_manual(y_test, nb_pred)
        knn_acc   = accuracy_score_manual(y_test, knn_pred)
        nb_wf1    = weighted_f1(nb_rep)
        knn_wf1   = weighted_f1(knn_rep)

        frame = self.full_metrics_frame


        self._section_label(frame, "Ringkasan Metrik Keseluruhan", CLR["primary"])

        def _best(nb_val, knn_val):
            if nb_val > knn_val:   return "NB", CLR["purple"]
            elif knn_val > nb_val: return "KNN", CLR["blue"]
            else:                  return "=", CLR["text_muted"]

        summary_rows = [
            ("Macro F1-Score (UTAMA)",  nb_rep["macro_avg"]["f1"],          knn_rep["macro_avg"]["f1"]),
            ("Weighted F1-Score",       nb_wf1,                             knn_wf1),
            ("Macro Precision",         nb_rep["macro_avg"]["precision"],    knn_rep["macro_avg"]["precision"]),
            ("Macro Recall",            nb_rep["macro_avg"]["recall"],       knn_rep["macro_avg"]["recall"]),
            ("Akurasi (info tambahan)", nb_acc,                             knn_acc),
        ]

        tbl = tk.Frame(frame, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        tbl.pack(fill="x", pady=(0, 10))
        hdr = tk.Frame(tbl, bg=CLR["primary_lt"])
        hdr.pack(fill="x")
        for txt, w in [("Metrik", 32), ("Naive Bayes", 18), ("KNN", 18), ("Unggul", 12)]:
            tk.Label(hdr, text=txt, width=w, font=("Courier New", 12, "bold"),
                     fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w").pack(side="left", padx=(6, 0), pady=3)
        tk.Frame(tbl, bg=CLR["border"], height=1).pack(fill="x")

        for i, (name, nb_v, knn_v) in enumerate(summary_rows):
            is_main = (i == 0)
            bg = "#FEF9C3" if is_main else (CLR["surface2"] if i % 2 else CLR["surface"])
            row = tk.Frame(tbl, bg=bg)
            row.pack(fill="x")
            winner, wcolor = _best(nb_v, knn_v)
            for txt, w, clr in [
                (name,          32, CLR["text_hd"] if is_main else CLR["text_body"]),
                (f"{nb_v:.4f}", 18, CLR["purple"]),
                (f"{knn_v:.4f}",18, CLR["blue"]),
                (winner,        12, wcolor),
            ]:
                tk.Label(row, text=txt, width=w,
                         font=("Courier New", 12, "bold" if is_main else "normal"),
                         fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=3)

        # Label pemenang
        nb_f1  = nb_rep["macro_avg"]["f1"]
        knn_f1 = knn_rep["macro_avg"]["f1"]
        if nb_f1 > knn_f1:
            winner_txt   = f"🏆  Naive Bayes unggul  (Macro F1: {nb_f1:.4f} vs {knn_f1:.4f})"
            winner_color = CLR["purple"]
        elif knn_f1 > nb_f1:
            winner_txt   = f"🏆  KNN unggul  (Macro F1: {knn_f1:.4f} vs {nb_f1:.4f})"
            winner_color = CLR["blue"]
        else:
            winner_txt   = f"🤝  Kedua model seimbang  (Macro F1: {nb_f1:.4f})"
            winner_color = CLR["success"]

        winner_box = tk.Frame(frame, bg=CLR["primary_lt"],
                              highlightbackground=winner_color, highlightthickness=2)
        winner_box.pack(fill="x", pady=(4, 0))
        tk.Label(winner_box, text=winner_txt,
                 font=("Segoe UI", 10, "bold"), fg=winner_color,
                 bg=CLR["primary_lt"], pady=8).pack()

        classes = sorted(set(y_test) | set(nb_pred) | set(knn_pred))

        for model_name, rep, accent in [
            ("Naive Bayes", nb_rep, CLR["purple"]),
            ("KNN",         knn_rep, CLR["blue"]),
        ]:
            self._section_label(frame, f"Detail Per Kelas — {model_name}", accent)
            ptbl = tk.Frame(frame, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
            ptbl.pack(fill="x", pady=(0, 10))
            phdr = tk.Frame(ptbl, bg=CLR["primary_lt"])
            phdr.pack(fill="x")
            for txt, w in [("Kelas", 26), ("Precision", 16), ("Recall", 16), ("F1-Score", 16), ("Support", 12)]:
                tk.Label(phdr, text=txt, width=w, font=("Courier New", 12, "bold"),
                         fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w").pack(side="left", padx=(6, 0), pady=3)
            tk.Frame(ptbl, bg=CLR["border"], height=1).pack(fill="x")

            for ci, cls in enumerate(classes):
                prec = rep["precision"].get(cls, 0.0)
                rec  = rep["recall"].get(cls, 0.0)
                f1   = rep["f1"].get(cls, 0.0)
                sup  = rep["support"].get(cls, 0)
                is_low_recall = (rec < 0.5 and sup > 0)
                bg = "#FEE2E2" if is_low_recall else (CLR["surface2"] if ci % 2 else CLR["surface"])
                row = tk.Frame(ptbl, bg=bg)
                row.pack(fill="x")
                warn = " ⚠" if is_low_recall else ""
                for txt, w in [
                    (str(cls) + warn, 26),
                    (f"{prec:.4f}", 16),
                    (f"{rec:.4f}",  16),
                    (f"{f1:.4f}",   16),
                    (str(sup),      12),
                ]:
                    tk.Label(row, text=txt, width=w, font=("Courier New", 12),
                             fg=CLR["danger"] if is_low_recall else CLR["text_body"],
                             bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

    # ─────────────────────────────────────────────────────────────────
    # ROW 2 — ERROR TABLE
    # ─────────────────────────────────────────────────────────────────
    def _build_error_table(self, parent):
        cols = ("no", "filename", "asli", "nb", "knn")
        self.error_tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        widths = {"no": 40, "filename": 260, "asli": 120, "nb": 120, "knn": 120}
        for c in cols:
            self.error_tree.heading(c, text=c.title())
            self.error_tree.column(c, width=widths.get(c, 120), anchor="w")
        self.error_tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(parent, command=self.error_tree.yview)
        sb.pack(side="right", fill="y")
        self.error_tree.configure(yscrollcommand=sb.set)

    def _populate_error_table(self, y_t, p_n, p_k, fn):
        for row in self.error_tree.get_children():
            self.error_tree.delete(row)
        err = 0
        for i, (yt, pn, pk) in enumerate(zip(y_t, p_n, p_k)):
            if pn != yt or pk != yt:
                err += 1
                fname = fn[i] if i < len(fn) else "N/A"
                self.error_tree.insert("", "end", values=(err, fname, yt, pn, pk))


    def _draw_single_fold_cm(self, parent, cm, labels, title, cmap, accent):
        n = len(labels)
        fig_size = max(3.8, n * 0.7)
        fig = Figure(figsize=(fig_size, fig_size), dpi=90)
        ax  = fig.add_subplot(111)

        ax.imshow(cm, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Prediksi", fontsize=9)
        ax.set_ylabel("Aktual",   fontsize=9)
        ax.set_title(title, fontsize=9, weight="bold", color=accent)

        max_val = cm.max() if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                val = int(cm[i, j])
                color = "white" if val > max_val * 0.5 else "black"
                ax.text(j, i, val, ha="center", va="center",
                        weight="bold", fontsize=9, color=color)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        # BUG FIX #3: fill="x" bukan fill="both" agar CM per fold tidak melar
        canvas.get_tk_widget().pack(fill="x")


    # ─────────────────────────────────────────────────────────────────
    # ROW 3 — K-FOLD CONTROL
    # ─────────────────────────────────────────────────────────────────
    def _setup_kfold_control_ui(self, card):
        self._info_box(card,
            "Stratified K-Fold memastikan proporsi kelas tetap seimbang di setiap fold,\n"
            "sehingga hasil evaluasi tidak bias terhadap kelas mayoritas.\n"
            "Standar deviasi antar fold menunjukkan stabilitas model — std rendah = model stabil.\n"
            "K-Fold dijalankan pada SELURUH dataset (train + test) sesuai standar evaluasi ML.",
            color=CLR["success"]
        )
        self.btn_run_kfold = tk.Button(
            card, text="Jalankan Stratified 5-Fold Cross Validation",
            command=self._run_kfold,
            bg=CLR["success"], fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat", pady=10
        )
        self.btn_run_kfold.pack(fill="x", pady=5)
        self.progress_kfold = ttk.Progressbar(card, mode="indeterminate")
        self.progress_kfold.pack(fill="x", pady=5)
        self.lbl_kfold_status = tk.Label(
            card, text="", font=("Segoe UI", 9, "italic"),
            fg=CLR["text_muted"], bg=CLR["surface"]
        )
        self.lbl_kfold_status.pack(anchor="w")

    # ─────────────────────────────────────────────────────────────────
    # ROW 4 — K-FOLD STATS TABLE
    # ─────────────────────────────────────────────────────────────────
    def _setup_kfold_stats_ui(self, card):
        cols = ("fold",
                "nb_acc",  "knn_acc",
                "nb_f1",   "knn_f1",
                "nb_prec", "knn_prec",
                "nb_rec",  "knn_rec",
                "nb_train_ms", "knn_pred_ms")
        tree_holder = tk.Frame(card, bg=CLR["surface"])
        tree_holder.pack(fill="x")

        self.kfold_tree = ttk.Treeview(tree_holder, columns=cols, show="headings", height=7)
        headings = {
            "fold":        "Fold",
            "nb_acc":      "Akurasi NB",   "knn_acc":     "Akurasi KNN",
            "nb_f1":       "Macro F1 NB",  "knn_f1":      "Macro F1 KNN",
            "nb_prec":     "Presisi NB",   "knn_prec":    "Presisi KNN",
            "nb_rec":      "Recall NB",    "knn_rec":     "Recall KNN",
            "nb_train_ms": "Train NB (ms)","knn_pred_ms": "Pred KNN (ms)",
        }
        W_FOLD, W_PAIR, W_TIME = 55, 100, 105
        widths = {
            "fold": W_FOLD,
            "nb_acc": W_PAIR,  "knn_acc":    W_PAIR,
            "nb_f1":  W_PAIR,  "knn_f1":     W_PAIR,
            "nb_prec":W_PAIR,  "knn_prec":   W_PAIR,
            "nb_rec": W_PAIR,  "knn_rec":    W_PAIR,
            "nb_train_ms": W_TIME, "knn_pred_ms": W_TIME,
        }
        for c in cols:
            self.kfold_tree.heading(c, text=headings[c])
            self.kfold_tree.column(c, width=widths[c], anchor="center", minwidth=widths[c])

        vsb = ttk.Scrollbar(tree_holder, orient="vertical",   command=self.kfold_tree.yview)
        xsb = ttk.Scrollbar(tree_holder, orient="horizontal", command=self.kfold_tree.xview)
        self.kfold_tree.configure(yscrollcommand=vsb.set, xscrollcommand=xsb.set)
        self.kfold_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        tree_holder.columnconfigure(0, weight=1)
        tree_holder.rowconfigure(0, weight=1)

        stat_outer = tk.Frame(card, bg=CLR["surface2"],
                              highlightbackground=CLR["border"], highlightthickness=1)
        stat_outer.pack(fill="x", pady=(10, 0))
        self.lbl_stats_kfold = tk.Label(
            stat_outer, text="Jalankan K-Fold CV untuk melihat statistik.",
            font=("Courier New", 9), justify="left",
            bg=CLR["surface2"], padx=12, pady=10, anchor="w"
        )
        self.lbl_stats_kfold.pack(fill="x")

    # ─────────────────────────────────────────────────────────────────
    # ROW 6 — CONFUSION MATRIX PER FOLD
    # ─────────────────────────────────────────────────────────────────
    def _setup_kfold_cm_ui(self, card):
        self.kfold_cm_frame = tk.Frame(card, bg=CLR["surface"])
        self.kfold_cm_frame.pack(fill="both", expand=True)
        tk.Label(
            self.kfold_cm_frame,
            text="Jalankan K-Fold CV untuk melihat confusion matrix tiap fold.",
            font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]
        ).pack(pady=16)

    # ─────────────────────────────────────────────────────────────────
    # ROW 7 — HOLDOUT vs CV COMPARISON
    # ─────────────────────────────────────────────────────────────────
    def _setup_holdout_cv_ui(self, card):
        self.holdout_cv_frame = card
        self._info_box(card,
            "Membandingkan performa holdout test (evaluasi final) dengan rata-rata K-Fold CV.\n"
            "Selisih besar (> 0.10) menandakan split data tidak representatif atau model tidak stabil.\n"
            "✓ = konsisten (≤ 0.05)   ⚠ = perlu diperiksa (0.05–0.10)   ✗ = tidak konsisten (> 0.10)",
            color=CLR["orange"]
        )
        tk.Label(card, text="Jalankan K-Fold CV dan Uji Model terlebih dahulu.",
                 font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"],
                 bg=CLR["surface"]).pack(pady=8)

    # ─────────────────────────────────────────────────────────────────
    # K-FOLD WORKER
    # ─────────────────────────────────────────────────────────────────
    def _run_kfold(self):
        if not self.modeling_frame:
            return messagebox.showwarning("Peringatan", "Modeling frame tidak tersedia.")
        if self.controller.X_train is None:
            return messagebox.showwarning("Peringatan", "Data belum siap. Lakukan preprocessing terlebih dahulu.")
        if not hasattr(self.controller, "df") or self.controller.df is None:
            return messagebox.showwarning("Peringatan", "Dataframe belum tersedia. Lakukan preprocessing terlebih dahulu.")

        self.btn_run_kfold.config(state="disabled")
        self.progress_kfold.start(10)
        self.lbl_kfold_status.config(text="Menjalankan K-Fold CV…")

        try:
            nb_alpha = self.modeling_frame.alpha
            knn_k    = self.modeling_frame.k
        except (ValueError, AttributeError):
            self.progress_kfold.stop()
            self.btn_run_kfold.config(state="normal")
            return messagebox.showerror("Input Tidak Valid", "Periksa nilai Alpha dan K di tab Modeling.")

        def worker():
            try:
                from feature_extraction import fit_transform_tfidf, transform_tfidf
                from modeling import ManualMultinomialNB, ManualKNN
                import time

                df     = self.controller.df
                texts  = df["text_clean"].fillna("").tolist()
                y_all  = np.array(df["label"].tolist())

                # Tentukan jumlah fold: max 5, min sesuai kelas terkecil
                counts = np.bincount(np.unique(y_all, return_inverse=True)[1])
                k_folds = min(5, int(counts.min()))
                if k_folds < 2:
                    raise ValueError(
                        f"K-Fold tidak dapat dilakukan: kelas terkecil hanya memiliki "
                        f"{counts.min()} sampel. Tambahkan data atau gunakan split berbeda."
                    )

                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                indices = np.arange(len(texts))

                nb_folds, knn_folds = [], []

                for fold_i, (tr_idx, val_idx) in enumerate(skf.split(indices, y_all)):
                    texts_tr  = [texts[i] for i in tr_idx]
                    texts_val = [texts[i] for i in val_idx]
                    y_tr  = y_all[tr_idx]
                    y_val = y_all[val_idx]

                    # TF-IDF fit pada fold train, transform fold val (no leakage)
                    X_tr, vec = fit_transform_tfidf(texts_tr)
                    X_val     = transform_tfidf(texts_val, vec)

                    # ── Naive Bayes ──────────────────────────────────
                    t0 = time.perf_counter()
                    nb = ManualMultinomialNB(alpha=nb_alpha)
                    nb.fit(X_tr, y_tr)
                    t_train_nb = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    nb_pred = nb.predict(X_val)
                    t_pred_nb = time.perf_counter() - t0

                    nb_rep, nb_cm, nb_labels = classification_report_manual(y_val, nb_pred)
                    nb_acc  = accuracy_score_manual(y_val, nb_pred)

                    nb_folds.append({
                        "accuracy":   nb_acc,
                        "macro_f1":   nb_rep["macro_avg"]["f1"],
                        "precision":  nb_rep["macro_avg"]["precision"],
                        "recall":     nb_rep["macro_avg"]["recall"],
                        "weighted_f1": weighted_f1(nb_rep),
                        "train_time": t_train_nb,
                        "pred_time":  t_pred_nb,
                        "cm":         nb_cm,
                        "cm_labels":  nb_labels,
                        "y_val":      y_val,
                        "y_pred":     nb_pred,
                    })

                    # ── KNN ──────────────────────────────────────────
                    t0 = time.perf_counter()
                    knn = ManualKNN(k=knn_k, metric="cosine")
                    knn.fit(X_tr, y_tr)
                    t_train_knn = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    knn_pred = knn.predict(X_val)
                    t_pred_knn = time.perf_counter() - t0

                    knn_rep, knn_cm, knn_labels = classification_report_manual(y_val, knn_pred)
                    knn_acc = accuracy_score_manual(y_val, knn_pred)

                    knn_folds.append({
                        "accuracy":   knn_acc,
                        "macro_f1":   knn_rep["macro_avg"]["f1"],
                        "precision":  knn_rep["macro_avg"]["precision"],
                        "recall":     knn_rep["macro_avg"]["recall"],
                        "weighted_f1": weighted_f1(knn_rep),
                        "train_time": t_train_knn,
                        "pred_time":  t_pred_knn,
                        "cm":         knn_cm,
                        "cm_labels":  knn_labels,
                        "y_val":      y_val,
                        "y_pred":     knn_pred,
                    })

                def _mean(folds, key):
                    return float(np.mean([f[key] for f in folds]))
                def _std(folds, key):
                    return float(np.std([f[key] for f in folds]))

                res = {
                    "k_folds": k_folds,
                    "nb": {
                        "folds":            nb_folds,
                        "mean_accuracy":    _mean(nb_folds, "accuracy"),
                        "std_accuracy":     _std(nb_folds,  "accuracy"),
                        "mean_macro_f1":    _mean(nb_folds, "macro_f1"),
                        "std_macro_f1":     _std(nb_folds,  "macro_f1"),
                        "mean_precision":   _mean(nb_folds, "precision"),
                        "mean_recall":      _mean(nb_folds, "recall"),
                        "mean_weighted_f1": _mean(nb_folds, "weighted_f1"),
                        "mean_train_time":  _mean(nb_folds, "train_time"),
                        "mean_pred_time":   _mean(nb_folds, "pred_time"),
                    },
                    "knn": {
                        "folds":            knn_folds,
                        "mean_accuracy":    _mean(knn_folds, "accuracy"),
                        "std_accuracy":     _std(knn_folds,  "accuracy"),
                        "mean_macro_f1":    _mean(knn_folds, "macro_f1"),
                        "std_macro_f1":     _std(knn_folds,  "macro_f1"),
                        "mean_precision":   _mean(knn_folds, "precision"),
                        "mean_recall":      _mean(knn_folds, "recall"),
                        "mean_weighted_f1": _mean(knn_folds, "weighted_f1"),
                        "mean_train_time":  _mean(knn_folds, "train_time"),
                        "mean_pred_time":   _mean(knn_folds, "pred_time"),
                    },
                }
                self.after(0, lambda r=res: self._update_kfold_ui(r))

            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error K-Fold", str(e)))
            finally:
                self.after(0, self.progress_kfold.stop)
                self.after(0, lambda: self.btn_run_kfold.config(state="normal"))
                self.after(0, lambda: self.lbl_kfold_status.config(text=""))

        threading.Thread(target=worker, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────
    # UPDATE K-FOLD UI
    # ─────────────────────────────────────────────────────────────────
    def _update_kfold_ui(self, res):
        self._last_cv_res = res

        # ── Tabel per-fold ────────────────────────────────────────────
        for row in self.kfold_tree.get_children():
            self.kfold_tree.delete(row)

        nb_folds  = res["nb"]["folds"]
        knn_folds = res["knn"]["folds"]

        for i, (nf, kf) in enumerate(zip(nb_folds, knn_folds)):
            self.kfold_tree.insert("", "end", values=(
                f"Fold {i+1}",
                f"{nf['accuracy']:.4f}",  f"{kf['accuracy']:.4f}",
                f"{nf['macro_f1']:.4f}",  f"{kf['macro_f1']:.4f}",
                f"{nf['precision']:.4f}", f"{kf['precision']:.4f}",
                f"{nf['recall']:.4f}",    f"{kf['recall']:.4f}",
                f"{nf['train_time']*1000:.2f}", f"{kf['pred_time']*1000:.2f}",
            ))

        # ── Label statistik ringkasan ─────────────────────────────────
        nb  = res["nb"]
        knn = res["knn"]
        k   = res["k_folds"]
        self.lbl_stats_kfold.config(text=(
            f"{'Metrik':<28}  {'Naive Bayes':>14}  {'KNN':>14}\n"
            f"{'─'*60}\n"
            f"{'Mean Akurasi':<28}  {nb['mean_accuracy']:>14.4f}  {knn['mean_accuracy']:>14.4f}\n"
            f"{'Std  Akurasi':<28}  {nb['std_accuracy']:>14.4f}  {knn['std_accuracy']:>14.4f}\n"
            f"{'Mean Macro F1':<28}  {nb['mean_macro_f1']:>14.4f}  {knn['mean_macro_f1']:>14.4f}\n"
            f"{'Std  Macro F1':<28}  {nb['std_macro_f1']:>14.4f}  {knn['std_macro_f1']:>14.4f}\n"
            f"{'Mean Presisi (Macro)':<28}  {nb['mean_precision']:>14.4f}  {knn['mean_precision']:>14.4f}\n"
            f"{'Mean Recall (Macro)':<28}  {nb['mean_recall']:>14.4f}  {knn['mean_recall']:>14.4f}\n"
            f"{'Mean Weighted F1':<28}  {nb['mean_weighted_f1']:>14.4f}  {knn['mean_weighted_f1']:>14.4f}\n"
            f"{'Mean Train Time (ms)':<28}  {nb['mean_train_time']*1000:>14.2f}  {knn['mean_train_time']*1000:>14.2f}\n"
            f"{'Mean Pred  Time (ms)':<28}  {nb['mean_pred_time']*1000:>14.2f}  {knn['mean_pred_time']*1000:>14.2f}\n"
            f"{'─'*60}\n"
            f"  Stratified {k}-Fold CV  ·  TF-IDF fit per fold (no data leakage)"
        ))

        # ── Grafik tren ───────────────────────────────────────────────
        self._draw_kfold_chart(
            [f["accuracy"]  for f in nb_folds],
            [f["accuracy"]  for f in knn_folds],
            [f["macro_f1"]  for f in nb_folds],
            [f["macro_f1"]  for f in knn_folds],
        )

        # ── CM per fold ───────────────────────────────────────────────
        self._draw_kfold_cm(res)

        # ── Holdout vs CV (jika holdout sudah ada) ────────────────────
        if self._y_test is not None and self._nb_pred is not None:
            self._render_holdout_cv_comparison(res)

    # ─────────────────────────────────────────────────────────────────
    # GRAFIK TREN PER FOLD
    # ─────────────────────────────────────────────────────────────────
    def _draw_kfold_chart(self, nb_acc, knn_acc, nb_f1, knn_f1):
        for w in self.kfold_chart_frame.winfo_children():
            w.destroy()

        fig  = Figure(figsize=(6.5, 3.8), dpi=95)
        folds = np.arange(1, len(nb_acc) + 1)

        ax1 = fig.add_subplot(211)
        ax1.plot(folds, nb_acc,  marker="o", label="Akurasi NB",  color=CLR["purple"], linewidth=2)
        ax1.plot(folds, knn_acc, marker="s", label="Akurasi KNN", color=CLR["blue"],   linewidth=2, linestyle="--")
        ax1.axhline(np.mean(nb_acc),  color=CLR["purple"], linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.axhline(np.mean(knn_acc), color=CLR["blue"],   linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.set_title("Tren Akurasi & Macro F1 Per Fold", fontsize=9, weight="bold")
        ax1.set_ylabel("Akurasi", fontsize=8)
        ax1.set_xticks(folds)
        ax1.legend(fontsize=7)
        ax1.set_facecolor(CLR["surface2"])

        ax2 = fig.add_subplot(212)
        ax2.plot(folds, nb_f1,  marker="o", label="Macro F1 NB",  color=CLR["purple"], linewidth=2)
        ax2.plot(folds, knn_f1, marker="s", label="Macro F1 KNN", color=CLR["blue"],   linewidth=2, linestyle="--")
        ax2.axhline(np.mean(nb_f1),  color=CLR["purple"], linewidth=0.8, linestyle=":", alpha=0.7)
        ax2.axhline(np.mean(knn_f1), color=CLR["blue"],   linewidth=0.8, linestyle=":", alpha=0.7)
        ax2.set_ylabel("Macro F1", fontsize=8)
        ax2.set_xlabel("Fold",     fontsize=8)
        ax2.set_xticks(folds)
        ax2.legend(fontsize=7)
        ax2.set_facecolor(CLR["surface2"])

        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=self.kfold_chart_frame)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────
    # CONFUSION MATRIX PER FOLD (Notebook tab)
    # ─────────────────────────────────────────────────────────────────
    def _draw_kfold_cm(self, res):
        for w in self.kfold_cm_frame.winfo_children():
            w.destroy()

        nb_folds  = res["nb"]["folds"]
        knn_folds = res["knn"]["folds"]
        n_folds   = len(nb_folds)

        style = ttk.Style()
        style.configure("KFoldCM.TNotebook.Tab", padding=[10, 4], font=("Segoe UI", 9))
        nb_widget = ttk.Notebook(self.kfold_cm_frame, style="KFoldCM.TNotebook")
        nb_widget.pack(fill="both", expand=True)

        for fold_idx in range(n_folds):
            fold_nb  = nb_folds[fold_idx]
            fold_knn = knn_folds[fold_idx]

            tab = tk.Frame(nb_widget, bg=CLR["surface"])
            nb_widget.add(tab, text=f"  Fold {fold_idx + 1}  ")
            tab.columnconfigure(0, weight=1, uniform="cmcol")
            tab.columnconfigure(1, weight=1, uniform="cmcol")

            # Header ringkasan fold
            hdr = tk.Frame(tab, bg=CLR["primary_lt"], padx=12, pady=6)
            hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(8, 4))
            tk.Label(hdr,
                text=(f"Fold {fold_idx+1}  │  "
                      f"NB: Acc={fold_nb['accuracy']:.4f}  Macro F1={fold_nb['macro_f1']:.4f}  │  "
                      f"KNN: Acc={fold_knn['accuracy']:.4f}  Macro F1={fold_knn['macro_f1']:.4f}"),
                font=("Consolas", 9, "bold"), bg=CLR["primary_lt"], fg=CLR["text_hd"]
            ).pack(anchor="w")

            # CM Naive Bayes
            nb_cm_frame = tk.Frame(tab, bg=CLR["surface"])
            nb_cm_frame.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=6)
            self._draw_single_fold_cm(
                nb_cm_frame, fold_nb["cm"], fold_nb["cm_labels"],
                f"Naive Bayes — Fold {fold_idx+1}", "Blues", CLR["purple"]
            )

            # CM KNN
            knn_cm_frame = tk.Frame(tab, bg=CLR["surface"])
            knn_cm_frame.grid(row=1, column=1, sticky="nsew", padx=(3, 6), pady=6)
            self._draw_single_fold_cm(
                knn_cm_frame, fold_knn["cm"], fold_knn["cm_labels"],
                f"KNN — Fold {fold_idx+1}", "Greens", CLR["blue"]
            )

    # ─────────────────────────────────────────────────────────────────
    # HOLDOUT vs CV COMPARISON
    # ─────────────────────────────────────────────────────────────────
    def _render_holdout_cv_comparison(self, cv_res):
        for w in self.holdout_cv_frame.winfo_children():
            w.destroy()

        frame = self.holdout_cv_frame
        self._info_box(frame,
            "Membandingkan performa holdout test (evaluasi final) dengan rata-rata K-Fold CV.\n"
            "Selisih besar (> 0.10) menandakan split data tidak representatif atau model tidak stabil.\n"
            "✓ = konsisten (≤ 0.05)   ⚠ = perlu diperiksa (0.05–0.10)   ✗ = tidak konsisten (> 0.10)",
            color=CLR["orange"]
        )

        nb_rep,  _, _ = classification_report_manual(self._y_test, self._nb_pred)
        knn_rep, _, _ = classification_report_manual(self._y_test, self._knn_pred)
        nb_acc_h  = accuracy_score_manual(self._y_test, self._nb_pred)
        knn_acc_h = accuracy_score_manual(self._y_test, self._knn_pred)

        def _flag(diff):
            a = abs(diff)
            if a <= 0.05:   return "✓", CLR["success"]
            elif a <= 0.10: return "⚠", CLR["warning"]
            else:           return "✗", CLR["danger"]

        for model_name, h_acc, h_rep, cv_key, accent in [
            ("Naive Bayes", nb_acc_h,  nb_rep,  "nb",  CLR["purple"]),
            ("KNN",         knn_acc_h, knn_rep, "knn", CLR["blue"]),
        ]:
            self._section_label(frame, f"— {model_name} —", accent)
            tbl = tk.Frame(frame, bg=CLR["surface"],
                           highlightbackground=CLR["border"], highlightthickness=1)
            tbl.pack(fill="x", pady=(0, 10))

            hdr = tk.Frame(tbl, bg=CLR["primary_lt"])
            hdr.pack(fill="x")
            for txt, w in [("Metrik", 26), ("Holdout Test", 16),
                           ("CV Mean ± Std", 22), ("Selisih", 12), ("Status", 8)]:
                tk.Label(hdr, text=txt, width=w, font=("Courier New", 9, "bold"),
                         fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                         ).pack(side="left", padx=(6, 0), pady=3)
            tk.Frame(tbl, bg=CLR["border"], height=1).pack(fill="x")

            cv  = cv_res[cv_key]
            folds = cv["folds"]
            metrics = [
                ("Akurasi",
                 h_acc,
                 cv["mean_accuracy"],
                 float(np.std([f["accuracy"]  for f in folds]))),
                ("Macro F1",
                 h_rep["macro_avg"]["f1"],
                 cv["mean_macro_f1"],
                 cv["std_macro_f1"]),
                ("Macro Presisi",
                 h_rep["macro_avg"]["precision"],
                 cv["mean_precision"],
                 float(np.std([f["precision"] for f in folds]))),
                ("Macro Recall",
                 h_rep["macro_avg"]["recall"],
                 cv["mean_recall"],
                 float(np.std([f["recall"]    for f in folds]))),
            ]

            for ri, (name, h_val, cv_mean, cv_std) in enumerate(metrics):
                diff = h_val - cv_mean
                flag_txt, flag_clr = _flag(diff)
                bg   = CLR["surface2"] if ri % 2 else CLR["surface"]
                sign = "+" if diff >= 0 else ""
                row  = tk.Frame(tbl, bg=bg)
                row.pack(fill="x")
                for txt, w, clr in [
                    (name,                            26, CLR["text_body"]),
                    (f"{h_val:.4f}",                  16, accent),
                    (f"{cv_mean:.4f} ± {cv_std:.4f}", 22, CLR["text_muted"]),
                    (f"{sign}{diff:.4f}",             12,
                     CLR["success"] if diff >= -0.005 else CLR["danger"]),
                    (flag_txt,                         8, flag_clr),
                ]:
                    tk.Label(row, text=txt, width=w, font=("Courier New", 9),
                             fg=clr, bg=bg, anchor="w"
                             ).pack(side="left", padx=(6, 0), pady=3)

    def update_from_test(self, y_test, nb_pred, knn_pred, filenames):
        self._y_test   = y_test
        self._nb_pred  = nb_pred
        self._knn_pred = knn_pred

        self._draw_matrix(y_test, nb_pred,  self.nb_matrix_frame,  "Blues",  "Naive Bayes")
        self._draw_matrix(y_test, knn_pred, self.knn_matrix_frame, "Greens", "KNN")

        self._render_full_metrics(y_test, nb_pred, knn_pred)
        self._populate_error_table(y_test, nb_pred, knn_pred, filenames)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")