import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import csr_matrix as _csr
from modeling import (ManualMultinomialNB, ManualKNN,
                      knn_compare_metrics)
from evaluation import accuracy_score_manual, classification_report_manual
from feature_extraction import fit_transform_tfidf, transform_tfidf

# Konfigurasi Warna
CLR = {
    "bg":          "#F0F2F8", "surface":     "#FFFFFF", "surface2":    "#F8FAFF", "primary":     "#4F46E5", "primary_dk":   "#3730A3", 
    "primary_lt":  "#EEF2FF", "success":     "#10B981", "warning":     "#F59E0B", "danger":      "#EF4444", "purple":       "#8B5CF6", 
    "blue":        "#3B82F6", "text_hd":     "#1E1B4B", "text_body":   "#374151", "text_muted":  "#6B7280", "border":       "#E5E7EB",
}

class ModelingFrame(tk.Frame):
    def __init__(self, parent, controller, on_results=None):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller
        self.on_results = on_results

        self.nb  = None
        self.knn = None

        # State attributes untuk Visualisasi Proses Modeling
        self.last_y_pred_nb       = None
        self.last_y_pred_knn      = None
        self.last_nb_acc          = None
        self.last_knn_acc         = None
        self.last_nb_report       = None
        self.last_knn_report      = None
        self.last_tuning_k_list   = None
        self.last_tuning_f1_scores = None
        self.last_alpha_results   = None
        self.last_metric_results  = None

        self._build_header()

        # Setup Canvas & Scrollbar
        self.canvas = tk.Canvas(self, bg=CLR["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=CLR["bg"], padx=28, pady=22)

        self._canvas_win = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfig(self._canvas_win, width=event.width))
        self.bind("<Map>",   lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.bind("<Unmap>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        self.scroll_frame.columnconfigure(0, weight=1, uniform="col")
        self.scroll_frame.columnconfigure(1, weight=1, uniform="col")

        # UI Components
        ctrl_card = self._create_control_card(self.scroll_frame, row=0, col=0, accent_color=CLR["primary"])
        self._setup_parameter_ui(ctrl_card)

        info_card = self._create_card(self.scroll_frame, "Informasi Model", row=0, col=1, accent_color=CLR["success"])
        self._setup_model_info_ui(info_card)

        nb_vis_card = self._create_card(self.scroll_frame, "Visualisasi — Naïve Bayes", row=1, col=0, columnspan=2, accent_color=CLR["purple"])
        self._setup_nb_vis_ui(nb_vis_card)

        knn_vis_card = self._create_card(self.scroll_frame, "Visualisasi — K-Nearest Neighbor", row=2, col=0, columnspan=2, accent_color=CLR["blue"])
        self._setup_knn_vis_ui(knn_vis_card)

    
    def _create_card(self, parent, title, row, col, columnspan=1, accent_color=CLR["primary"]):
        outer = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.grid(row=row, column=col, columnspan=columnspan, sticky="nsew", padx=10, pady=(0, 20))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=18, pady=12)
        hd.pack(fill="x")
        tk.Label(hd, text=f"{title}", font=("Segoe UI", 11, "bold"), bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=20)
        content.pack(fill="both", expand=True)
        return content

    def _create_control_card(self, parent, row, col, columnspan=1, accent_color=CLR["primary"]):
        card = tk.Frame(parent, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        card.grid(row=row, column=col, columnspan=columnspan, sticky="nsew", padx=10, pady=(0, 20))
        tk.Frame(card, bg=accent_color, height=4).pack(fill="x")
        content = tk.Frame(card, bg=CLR["surface"], padx=20, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"], highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(side="top", fill="x")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="both", expand=True)
        tk.Label(inner, text="Tahap 6: Pembangunan Model", font=("Segoe UI", 18, "bold"), bg=CLR["surface"], fg=CLR["text_hd"]).pack(anchor="w")

    def _setup_parameter_ui(self, card):
        # Alpha ditetapkan tetap 1.0 (Laplace Smoothing)
        tk.Label(card, text="Alpha (NB):", bg=CLR["surface"]).grid(row=1, column=0, sticky="w", pady=5)
        alpha_frame = tk.Frame(card, bg=CLR["surface"])
        alpha_frame.grid(row=1, column=1, sticky="e")
        tk.Label(alpha_frame, text="1.0", font=("Segoe UI", 10, "bold"),
                 fg=CLR["primary"], bg=CLR["surface"]).pack(side="left")
        tk.Label(alpha_frame, text=" (Laplace Smoothing)",
                 font=("Segoe UI", 8), fg=CLR["text_muted"], bg=CLR["surface"]).pack(side="left")

        tk.Label(card, text="K-Neighbors:", bg=CLR["surface"]).grid(row=2, column=0, sticky="w", pady=5)
        k_frame = tk.Frame(card, bg=CLR["surface"])
        k_frame.grid(row=2, column=1, sticky="e")
        self.k_var = tk.IntVar(value=5)
        for k_val in [1, 3, 5, 7]:
            tk.Radiobutton(k_frame, text=str(k_val), variable=self.k_var, value=k_val,
                           bg=CLR["surface"], fg=CLR["text_body"],
                           activebackground=CLR["surface"],
                           selectcolor=CLR["primary_lt"],
                           font=("Segoe UI", 9, "bold")).pack(side="left", padx=4)

        self.btn_tune = tk.Button(card, text="Cari K Terbaik (K = 1, 3, 5, 7)", command=self._run_tuning,
                                  bg=CLR["blue"], fg="white",
                                  font=("Segoe UI", 9, "bold"), relief="flat", pady=8)
        self.btn_tune.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        self.btn_train = tk.Button(card, text="Latih Model", command=self._train_model,
                                   bg=CLR["success"], fg="white",
                                   font=("Segoe UI", 9, "bold"), relief="flat", pady=8)
        self.btn_train.grid(row=7, column=0, columnspan=2, sticky="ew", pady=5)

        self.btn_test = tk.Button(card, text="Uji Model", command=self._run_test,
                                  bg=CLR["primary"], fg="white",
                                  font=("Segoe UI", 9, "bold"), relief="flat", pady=8, state="disabled")
        self.btn_test.grid(row=9, column=0, columnspan=2, sticky="ew", pady=5)

        self.progress = ttk.Progressbar(card, mode="indeterminate")
        self.progress.grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)

        tk.Label(card,
                 text="Alpha = 1.0 ditetapkan berdasarkan Laplace Smoothing (standar literatur).\n"
                      "K dipilih dari bilangan ganjil 1–7. Evaluasi utama tetap menggunakan holdout test dan K-Fold di menu Analysis.",
                 font=("Segoe UI", 8), fg=CLR["text_muted"], bg=CLR["surface"],
                 wraplength=260, justify="left").grid(row=11, column=0, columnspan=2, sticky="w", pady=(0, 8))

    def _setup_model_info_ui(self, card):
        self.model_info_frame = tk.Frame(card, bg=CLR["surface"])
        self.model_info_frame.pack(fill="both", expand=True)

    def _run_tuning(self):
        if self.controller.X_train is None:
            return messagebox.showwarning("Peringatan", "Data belum siap.")
        if not hasattr(self.controller, "y_train") or self.controller.y_train is None:
            return messagebox.showwarning("Peringatan", "Label train (y_train) belum tersedia. Lakukan preprocessing terlebih dahulu.")
        if not hasattr(self.controller, "train_idx"):
            return messagebox.showwarning("Peringatan", "Indeks train (train_idx) belum tersedia. Lakukan split data terlebih dahulu.")
        if not hasattr(self.controller, "df") or self.controller.df is None:
            return messagebox.showwarning("Peringatan", "Dataframe belum tersedia.")
        self.progress.start(10)
        self.btn_tune.config(state="disabled")

        def worker():
            try:
                df          = self.controller.df
                tr_idx      = list(self.controller.train_idx)
                texts_all   = df["text_clean"].fillna("").tolist()
                texts_train = [texts_all[i] for i in tr_idx]
                y           = self.controller.y_train

                min_class_count = int(np.min(np.unique(y, return_counts=True)[1]))
                # Pilih LOOCV otomatis jika kelas terkecil < 3 (tidak cukup untuk fold)
                use_loocv = min_class_count < 3
                k_list    = [1, 3, 5, 7]

                if use_loocv:
                    # ── LOOCV: evaluasi tiap K dengan leave-one-out pada data train ──
                    from modeling import knn_search_k
                    X_tr_full, _ = fit_transform_tfidf(texts_train)
                    tuning_res   = knn_search_k(
                        X_tr_full, y, None, None,
                        k_list=k_list, use_loocv=True,
                    )
                    best_k    = tuning_res["best_k"]
                    best_f1   = tuning_res["best_f1"]
                    f1_scores = tuning_res["f1_list"]
                    mode_str  = f"LOOCV (n={tuning_res['loocv_n']})"

                    # Untuk perbandingan metrik: gunakan split leave-out-first
                    n_tr    = len(texts_train)
                    idx_tr0 = list(range(1, n_tr))
                    idx_v0  = [0]

                else:
                    # ── StratifiedKFold: jalur standar ──────────────────────────
                    n_splits = min(5, min_class_count)
                    if n_splits < 2:
                        raise ValueError(
                            "Tuning tidak dapat dilakukan: setiap kelas harus memiliki "
                            "minimal 2 sampel. Tambah data atau gunakan Skenario B."
                        )

                    from sklearn.model_selection import StratifiedKFold
                    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    indices   = np.arange(len(texts_train))
                    all_folds = list(skf.split(indices, y))

                    f1_scores = []
                    best_k, best_f1 = k_list[0], -1.0
                    for k in k_list:
                        fold_f1s = []
                        for fold_tr, fold_val in all_folds:
                            X_tr, vec = fit_transform_tfidf([texts_train[i] for i in fold_tr])
                            X_val     = transform_tfidf([texts_train[i] for i in fold_val], vec)
                            knn       = ManualKNN(k=k)
                            knn.fit(X_tr, y[fold_tr])
                            y_pred    = knn.predict(X_val)
                            rep, _, _ = classification_report_manual(y[fold_val], y_pred)
                            fold_f1s.append(rep["macro_avg"]["f1"])
                        mean_f1 = float(np.mean(fold_f1s))
                        f1_scores.append(mean_f1)
                        if mean_f1 > best_f1:
                            best_f1, best_k = mean_f1, k

                    mode_str        = f"StratifiedKFold (k={n_splits})"
                    idx_tr0, idx_v0 = all_folds[0]

                # Perbandingan metrik jarak (konfirmasi teoritis pada satu split)
                X_tr0, vec0 = fit_transform_tfidf([texts_train[i] for i in idx_tr0])
                X_val0      = transform_tfidf([texts_train[i] for i in idx_v0], vec0)
                metric_results = knn_compare_metrics(
                    X_tr0, y[np.array(idx_tr0)], X_val0, y[np.array(idx_v0)],
                    best_k, classification_report_manual,
                )

                def on_done(bk=best_k, bf=best_f1, kl=k_list, fs=f1_scores,
                            mr=metric_results, ms=mode_str):
                    self.k_var.set(bk)
                    self.last_tuning_k_list    = kl
                    self.last_tuning_f1_scores = fs
                    self.last_alpha_results    = None
                    self.last_metric_results   = mr
                    self.last_tuning_mode      = ms
                    self.btn_tune.config(text=f"Cari K Terbaik — {ms}")
                    messagebox.showinfo(
                        "Tuning Selesai",
                        f"Metode   : {ms}\n"
                        f"K terbaik: {bk}\n"
                        f"Macro F1 : {bf:.4f}\n\n"
                        f"K telah diset otomatis ke {bk}."
                    )

                self.after(0, on_done)

            except Exception as exc:
                self.after(0, lambda err=exc: messagebox.showerror("Error Tuning", str(err)))
            finally:
                self.after(0, self.progress.stop)
                self.after(0, lambda: self.btn_tune.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _train_model(self):
        if self.controller.X_train is None:
            return messagebox.showwarning("Peringatan", "Data belum siap.")
        try:
            alpha = 1.0
            k = int(self.k_var.get())
            if k < 1:
                return messagebox.showerror("Input Tidak Valid", "K harus bernilai minimal 1.")
        except ValueError:
            return messagebox.showerror("Input Tidak Valid", "Alpha harus berupa desimal dan K harus berupa bilangan bulat.")

        self.progress.start(10)
        self.btn_train.config(state="disabled")
        self.btn_test.config(state="disabled")

        def worker():
            try:
                nb = ManualMultinomialNB(alpha=alpha).fit(self.controller.X_train, self.controller.y_train)
                knn = ManualKNN(k=k).fit(self.controller.X_train, self.controller.y_train)
                self.nb = nb
                self.knn = knn
                self.controller.nb_model  = nb
                self.controller.knn_model = knn

                # Simpan nama file train ke controller agar _render_knn_calculations
                # dapat menampilkan nama file di tabel tetangga terdekat.
                if not hasattr(self.controller, "train_filenames") or self.controller.train_filenames is None:
                    df = getattr(self.controller, "df", None)
                    tr_idx = getattr(self.controller, "train_idx", None)
                    if df is not None and tr_idx is not None and "filename" in df.columns:
                        self.controller.train_filenames = [df["filename"].iloc[i] for i in tr_idx]
                    else:
                        self.controller.train_filenames = []

                def on_done():
                    self.btn_test.config(state="normal")
                    self._update_model_info(alpha, k)
                    messagebox.showinfo("Sukses", f"Model berhasil dilatih.\nNB alpha={alpha}, KNN k={k}")

                self.after(0, on_done)

            except Exception as exc:
                self.after(0, lambda err=exc: messagebox.showerror("Error Pelatihan", str(err)))
            finally:
                self.after(0, self.progress.stop)
                self.after(0, lambda: self.btn_train.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_model_info(self, alpha, k):
        for w in self.model_info_frame.winfo_children():
            w.destroy()
        info_lines = [
            ("Model", "Naive Bayes  &  KNN"),
            ("NB Alpha", str(alpha)),
            ("KNN (k)", str(k)),
            ("Fitur Train", str(self.controller.X_train.shape[1])),
            ("Sampel Train", str(self.controller.X_train.shape[0])),
            ("Sampel Test", str(self.controller.X_test.shape[0])),
        ]
        for label, value in info_lines:
            row_f = tk.Frame(self.model_info_frame, bg=CLR["surface"])
            row_f.pack(fill="x", pady=3)
            tk.Label(row_f, text=f"{label}:", width=14, anchor="w", bg=CLR["surface"],
                     fg=CLR["text_muted"], font=("Segoe UI", 9)).pack(side="left")
            tk.Label(row_f, text=value, anchor="w", bg=CLR["surface"],
                     fg=CLR["text_body"], font=("Segoe UI", 9, "bold")).pack(side="left")

    def _run_test(self):
        if not self.nb or not self.knn:
            return messagebox.showwarning("Peringatan", "Latih model terlebih dahulu.")
        if not hasattr(self.controller, "y_test") or self.controller.y_test is None:
            return messagebox.showwarning("Peringatan", "Label test (y_test) belum tersedia. Lakukan split data terlebih dahulu.")
        self.progress.start(10)
        self.btn_test.config(state="disabled")

        def worker():
            try:
                y_p_nb  = self.nb.predict(self.controller.X_test)
                y_p_knn = self.knn.predict(self.controller.X_test)
                nb_acc  = accuracy_score_manual(self.controller.y_test, y_p_nb)
                knn_acc = accuracy_score_manual(self.controller.y_test, y_p_knn)
                nb_rep,  _, _ = classification_report_manual(self.controller.y_test, y_p_nb)
                knn_rep, _, _ = classification_report_manual(self.controller.y_test, y_p_knn)

                # Evaluasi di data latih untuk deteksi overfitting
                nb_train_pred  = self.nb.predict(self.controller.X_train)
                knn_train_pred = self.knn.predict(self.controller.X_train)
                nb_train_acc   = accuracy_score_manual(self.controller.y_train, nb_train_pred)
                knn_train_acc  = accuracy_score_manual(self.controller.y_train, knn_train_pred)
                nb_train_rep,  _, _ = classification_report_manual(self.controller.y_train, nb_train_pred)
                knn_train_rep, _, _ = classification_report_manual(self.controller.y_train, knn_train_pred)

                def on_done(na=nb_acc, ka=knn_acc, nr=nb_rep, kr=knn_rep, pn=y_p_nb, pk=y_p_knn,
                            nta=nb_train_acc, kta=knn_train_acc,
                            ntr=nb_train_rep, ktr=knn_train_rep):
                    self._update_metrics(na, ka, nr, kr, pn, pk, nta, kta, ntr, ktr)

                self.after(0, on_done)

            except Exception as exc:
                # FIX: Capture exception secara eksplisit
                self.after(0, lambda err=exc: messagebox.showerror("Error Pengujian", str(err)))
            finally:
                self.after(0, self.progress.stop)
                self.after(0, lambda: self.btn_test.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_metrics(self, nb_a, knn_a, nb_r, knn_r, y_p_n, y_p_k,
                        nb_train_a=None, knn_train_a=None,
                        nb_train_r=None, knn_train_r=None):
        # Simpan state hasil uji
        self.last_y_pred_nb   = y_p_n
        self.last_y_pred_knn  = y_p_k
        self.last_nb_acc      = nb_a
        self.last_knn_acc     = knn_a
        self.last_nb_report   = nb_r
        self.last_knn_report  = knn_r
        self.last_nb_train_acc   = nb_train_a
        self.last_knn_train_acc  = knn_train_a
        self.last_nb_train_rep   = nb_train_r
        self.last_knn_train_rep  = knn_train_r

        # Render visualisasi NB dan KNN
        self._render_nb_vis()
        self._render_knn_vis()

        if self.on_results:
            self.on_results(
                self.controller.y_test, y_p_n, y_p_k,
                getattr(self.controller, "test_filenames", [])
            )

    # CONTOH PERHITUNGAN NB
    # ======================================================
    #   HELPER: KARTU AKURASI LATIH VS UJI
    # ======================================================

    def _render_train_test_card(self, parent, model_name, train_acc, test_acc,
                                train_rep, test_rep, accent_color):

        gap   = train_acc - test_acc
        mac_tr = train_rep["macro_avg"]["f1"] if train_rep else None
        mac_te = test_rep["macro_avg"]["f1"]  if test_rep  else None
        f1_gap = (mac_tr - mac_te) if (mac_tr is not None and mac_te is not None) else None

        if gap > 0.15:
            status, status_clr = "⚠  Indikasi Overfitting", CLR["danger"]
            status_tip = "Akurasi latih jauh lebih tinggi dari uji. Model terlalu hafal data latih."
        elif gap < -0.05:
            status, status_clr = "⚠  Indikasi Underfitting", CLR["warning"]
            status_tip = "Akurasi uji lebih tinggi dari latih — tidak biasa, periksa data."
        else:
            status, status_clr = "✓  Generalisasi Baik", CLR["success"]
            status_tip = "Selisih akurasi latih dan uji kecil — model cukup generalisatif."

        outer = tk.Frame(parent, bg=CLR["surface"],
                         highlightbackground=accent_color, highlightthickness=1)
        outer.pack(fill="x", pady=(0, 12))

        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=10, pady=6)
        hd.pack(fill="x")
        tk.Label(hd, text=f"Metrik Data Latih vs Data Uji — {model_name}",
                 font=("Segoe UI", 10, "bold"), fg=accent_color,
                 bg=CLR["primary_lt"]).pack(side="left")
        tk.Label(hd, text=status, font=("Segoe UI", 9, "bold"),
                 fg=status_clr, bg=CLR["primary_lt"]).pack(side="right")
        tk.Frame(outer, bg=accent_color, height=2).pack(fill="x")

        body = tk.Frame(outer, bg=CLR["surface"], padx=12, pady=10)
        body.pack(fill="x")

        # ── Baris 1 KPI: Akurasi ────────────────────────────────────
        row1 = tk.Frame(body, bg=CLR["surface"])
        row1.pack(fill="x", pady=(0, 6))
        for label, val, bg, fg in [
            ("Akurasi Data Latih",  f"{train_acc:.4f}", "#ECFDF5",         CLR["success"]),
            ("Akurasi Data Uji",    f"{test_acc:.4f}",  CLR["primary_lt"], accent_color),
            ("Selisih Akurasi",     f"{gap:+.4f}",
             "#FFFBEB" if abs(gap) > 0.15 else CLR["surface2"],
             CLR["danger"] if gap > 0.15 else CLR["text_body"]),
        ]:
            kf = tk.Frame(row1, bg=bg, highlightbackground=CLR["border"], highlightthickness=1)
            kf.pack(side="left", fill="both", expand=True, padx=(0, 8), ipadx=10, ipady=8)
            tk.Label(kf, text=val,   font=("Segoe UI", 16, "bold"), fg=fg, bg=bg).pack()
            tk.Label(kf, text=label, font=("Segoe UI", 8),          fg=fg, bg=bg).pack()

        # ── Baris 2 KPI: Macro F1 ───────────────────────────────────
        row2 = tk.Frame(body, bg=CLR["surface"])
        row2.pack(fill="x", pady=(0, 8))
        for label, val, bg, fg in [
            ("Macro F1 Latih",   f"{mac_tr:.4f}" if mac_tr is not None else "—",
             "#ECFDF5", CLR["success"]),
            ("Macro F1 Uji",     f"{mac_te:.4f}" if mac_te is not None else "—",
             CLR["primary_lt"], accent_color),
            ("Selisih Macro F1", f"{f1_gap:+.4f}" if f1_gap is not None else "—",
             "#FFFBEB" if (f1_gap is not None and abs(f1_gap) > 0.15) else CLR["surface2"],
             CLR["danger"] if (f1_gap is not None and f1_gap > 0.15) else CLR["text_body"]),
        ]:
            kf = tk.Frame(row2, bg=bg, highlightbackground=CLR["border"], highlightthickness=1)
            kf.pack(side="left", fill="both", expand=True, padx=(0, 8), ipadx=10, ipady=8)
            tk.Label(kf, text=val,   font=("Segoe UI", 14, "bold"), fg=fg, bg=bg).pack()
            tk.Label(kf, text=label, font=("Segoe UI", 8),          fg=fg, bg=bg).pack()

        # Tip
        tk.Label(body, text=f"ℹ  {status_tip}",
                 font=("Segoe UI", 8, "italic"), fg=status_clr,
                 bg=CLR["surface"]).pack(anchor="w", pady=(0, 10))

        # ── Tabel metrik per kelas ───────────────────────────────────
        for section_label, rep, side_color in [
            ("Detail Metrik per Kelas — Data Latih", train_rep, CLR["success"]),
            ("Detail Metrik per Kelas — Data Uji",   test_rep,  accent_color),
        ]:
            if rep is None:
                continue

            sec_f = tk.Frame(body, bg=CLR["surface"])
            sec_f.pack(fill="x", pady=(8, 4))
            tk.Frame(sec_f, bg=side_color, width=4).pack(side="left", fill="y")
            tk.Label(sec_f, text=f"  {section_label}",
                     font=("Segoe UI", 9, "bold"), fg=side_color,
                     bg=CLR["surface"]).pack(side="left", anchor="w")

            tbl = tk.Frame(body, bg=CLR["surface"],
                           highlightbackground=CLR["border"], highlightthickness=1)
            tbl.pack(fill="x", pady=(0, 6))

            hdr_f = tk.Frame(tbl, bg=CLR["primary_lt"])
            hdr_f.pack(fill="x")
            for txt, w in [("Kelas", 24), ("Precision", 16), ("Recall", 16),
                           ("F1-Score", 16), ("Support", 12)]:
                tk.Label(hdr_f, text=txt, width=w, font=("Courier New", 12, "bold"),
                         fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                         ).pack(side="left", padx=(6, 0), pady=3)
            tk.Frame(tbl, bg=CLR["border"], height=1).pack(fill="x")

            for ci, cls in enumerate(sorted(rep["precision"].keys())):
                prec   = rep["precision"].get(cls, 0.0)
                rec    = rep["recall"].get(cls, 0.0)
                f1     = rep["f1"].get(cls, 0.0)
                sup    = rep["support"].get(cls, 0)
                is_low = (rec < 0.5 and sup > 0)
                bg     = "#FEE2E2" if is_low else (CLR["surface2"] if ci % 2 else CLR["surface"])
                warn   = " ⚠" if is_low else ""
                row_f  = tk.Frame(tbl, bg=bg)
                row_f.pack(fill="x")
                for txt, w, clr in [
                    (str(cls) + warn, 24, CLR["danger"] if is_low else CLR["text_body"]),
                    (f"{prec:.4f}",   16, CLR["text_body"]),
                    (f"{rec:.4f}",    16, CLR["danger"] if is_low else CLR["text_body"]),
                    (f"{f1:.4f}",     16, side_color),
                    (str(sup),        12, CLR["text_muted"]),
                ]:
                    tk.Label(row_f, text=txt, width=w, font=("Courier New", 12),
                             fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

            # Baris Macro Avg saja
            hf = tk.Frame(tbl, bg=CLR["primary_lt"])
            hf.pack(fill="x")
            for txt, w in [
                ("Macro Avg",                           24),
                (f"{rep['macro_avg']['precision']:.4f}", 16),
                (f"{rep['macro_avg']['recall']:.4f}",    16),
                (f"{rep['macro_avg']['f1']:.4f}",        16),
                ("—",                                   12),
            ]:
                tk.Label(hf, text=txt, width=w, font=("Courier New", 12, "bold"),
                         fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                         ).pack(side="left", padx=(6, 0), pady=3)

        # ── Bar chart: Akurasi & Macro F1 ───────────────────────────
        fig  = Figure(figsize=(6, 2.6), dpi=100)
        ax   = fig.add_subplot(111)
        x    = np.arange(2)
        w    = 0.28

        for (mlabel, vals, color, alpha), offset in zip(
            [
                ("Akurasi",  [train_acc,       test_acc],       CLR["success"],  0.9),
                ("Macro F1", [mac_tr or 0,     mac_te or 0],    accent_color,    0.8),
            ],
            [-w / 2, w / 2],
        ):
            bars = ax.bar(x + offset, vals, width=w, color=color,
                          alpha=alpha, label=mlabel, edgecolor="white")
            ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(["Data Latih", "Data Uji"], fontsize=9)
        ax.set_title(f"{model_name} — Akurasi & Macro F1: Latih vs Uji",
                     fontsize=9, weight="bold")
        ax.set_ylabel("Nilai", fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_facecolor(CLR["surface2"])
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=body)
        cv.draw()
        cv.get_tk_widget().pack(fill="x", pady=(8, 0))

    # ======================================================
    #   VISUALISASI NB
    # ======================================================

    def _setup_nb_vis_ui(self, card):
        self.nb_vis_card = card
        tk.Label(card, text="Latih model terlebih dahulu untuk melihat visualisasi Naïve Bayes.",
                 font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=20)

    def _render_nb_vis(self):
        for w in self.nb_vis_card.winfo_children():
            w.destroy()
        parent = self.nb_vis_card
        nb     = self.nb
        ctrl   = self.controller

        # Kartu Alpha
        card_row = tk.Frame(parent, bg=CLR["surface"])
        card_row.pack(fill="x", pady=(0, 8))
        alpha_card = tk.Frame(card_row, bg=CLR["primary_lt"],
                              highlightbackground=CLR["purple"], highlightthickness=1)
        alpha_card.pack(side="left", padx=(0, 12), pady=4, ipadx=14, ipady=8)
        tk.Label(alpha_card, text="Alpha (α)", font=("Segoe UI", 8),
                 fg=CLR["text_muted"], bg=CLR["primary_lt"]).pack()
        tk.Label(alpha_card, text="1.0 (Laplace)", font=("Segoe UI", 13, "bold"),
                 fg=CLR["purple"], bg=CLR["primary_lt"]).pack()

        # ── Kartu Akurasi Latih vs Uji ───────────────────────────────
        nb_train_a = getattr(self, "last_nb_train_acc", None)
        nb_test_a  = getattr(self, "last_nb_acc",       None)
        nb_train_r = getattr(self, "last_nb_train_rep", None)

        if nb_train_a is not None and nb_test_a is not None:
            self._render_train_test_card(
                parent, "Naïve Bayes", nb_train_a, nb_test_a,
                nb_train_r, getattr(self, "last_nb_report", None),
                CLR["purple"],
            )

        if nb is None:
            tk.Label(parent, text="Latih model terlebih dahulu.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=8)
            return

        # Tabel prior kelas
        tk.Label(parent, text="Prior Tiap Kelas", font=("Segoe UI", 10, "bold"),
                 fg=CLR["purple"], bg=CLR["surface"]).pack(anchor="w", pady=(6, 2))

        tbl_f = tk.Frame(parent, bg=CLR["surface"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        tbl_f.pack(fill="x", pady=(0, 8))

        hdr = tk.Frame(tbl_f, bg=CLR["primary_lt"])
        hdr.pack(fill="x")
        for txt, w in [("Kelas", 24), ("Sampel", 14), ("P(C)", 18), ("log P(C)", 18)]:
            tk.Label(hdr, text=txt, width=w, font=("Courier New", 12, "bold"),
                     fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w").pack(side="left", padx=(6, 0), pady=3)
        tk.Frame(tbl_f, bg=CLR["border"], height=1).pack(fill="x")

        for i, c in enumerate(nb.classes_):
            lp     = float(nb.class_log_prior_[c])
            pr     = float(np.exp(lp))
            n_samp = nb.n_class_samples_.get(c, "-") if hasattr(nb, "n_class_samples_") else "-"
            bg     = CLR["surface2"] if i % 2 else CLR["surface"]
            row_f  = tk.Frame(tbl_f, bg=bg)
            row_f.pack(fill="x")
            for txt, w in [(str(c), 24), (str(n_samp), 14), (f"{pr:.6f}", 18), (f"{lp:.6f}", 18)]:
                tk.Label(row_f, text=txt, width=w, font=("Courier New", 12),
                         fg=CLR["text_body"], bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

        # Grafik sampel per kelas
        classes = list(nb.classes_)
        counts  = [nb.n_class_samples_.get(c, 0) for c in classes] if hasattr(nb, "n_class_samples_") else []
        if counts:
            self._draw_simple_bar_chart(parent, [str(c) for c in classes], counts,
                                        "Jumlah Sampel per Kelas", "Sampel", CLR["purple"])

        # Top-5 fitur per kelas
        vectorizer = getattr(ctrl, "vectorizer", None)
        if vectorizer is not None:
            vocab = vectorizer.get_feature_names_out()
            tk.Label(parent, text="Top-5 Kata Dominan per Kelas",
                     font=("Segoe UI", 10, "bold"), fg=CLR["purple"], bg=CLR["surface"]).pack(anchor="w", pady=(8, 2))
            top_cont = tk.Frame(parent, bg=CLR["surface"])
            top_cont.pack(fill="x", pady=(0, 8))
            for ci, c in enumerate(nb.classes_):
                col_f = tk.Frame(top_cont, bg=CLR["surface"],
                                 highlightbackground=CLR["border"], highlightthickness=1)
                col_f.grid(row=0, column=ci, sticky="nsew", padx=(0 if ci == 0 else 6, 0))
                top_cont.columnconfigure(ci, weight=1)
                tk.Label(col_f, text=str(c), font=("Segoe UI", 8, "bold"),
                         fg=CLR["purple"], bg=CLR["primary_lt"], pady=4).pack(fill="x")
                flp     = nb.feature_log_prob_[c]
                top_idx = np.argsort(flp)[-5:][::-1]
                for idx in top_idx:
                    word  = vocab[idx] if idx < len(vocab) else f"fitur-{idx}"
                    score = float(flp[idx])
                    row_f = tk.Frame(col_f, bg=CLR["surface"])
                    row_f.pack(fill="x")
                    tk.Label(row_f, text=f"  {word}", font=("Courier New", 11),
                             fg=CLR["text_body"], bg=CLR["surface"], anchor="w").pack(side="left")
                    tk.Label(row_f, text=f"{score:.3f}  ", font=("Courier New", 11),
                             fg=CLR["text_muted"], bg=CLR["surface"], anchor="e").pack(side="right")

        # ── Detail Perhitungan NB per Dokumen Uji (dropdown) ─────────
        X_test         = getattr(ctrl, "X_test", None)
        test_filenames = getattr(ctrl, "test_filenames", [])
        y_test         = getattr(ctrl, "y_test", None)

        if X_test is not None and X_test.shape[0] > 0 and vectorizer is not None:
            tk.Frame(parent, bg=CLR["border"], height=1).pack(fill="x", pady=(10, 8))
            tk.Label(parent,
                     text="Detail Perhitungan Naïve Bayes per Dokumen Uji",
                     font=("Segoe UI", 10, "bold"), fg=CLR["purple"], bg=CLR["surface"]
                     ).pack(anchor="w", pady=(0, 4))

            # Formula ringkas
            fml_box = tk.Frame(parent, bg="#F5F3FF",
                               highlightbackground=CLR["purple"], highlightthickness=1)
            fml_box.pack(fill="x", pady=(0, 8))
            tk.Label(fml_box,
                     text="  Formula:  log P(C|d) = log P(C) + Σ tfidf(tᵢ) × log P(tᵢ|C)\n"
                          "  Kelas dengan log P(C|d) tertinggi dipilih sebagai prediksi.",
                     font=("Courier New", 11), fg=CLR["text_body"],
                     bg="#F5F3FF", padx=10, pady=6, justify="left").pack(anchor="w")

            n_test = X_test.shape[0]

            def _nb_doc_label(i):
                fn  = test_filenames[i] if i < len(test_filenames) else f"Dokumen Uji #{i+1}"
                lbl = y_test[i] if y_test is not None and i < len(y_test) else "?"
                return f"[{i+1}] {fn}  (label: {lbl})"

            nb_doc_options   = [_nb_doc_label(i) for i in range(n_test)]
            self._nb_doc_var = tk.StringVar(value=nb_doc_options[0])

            nb_dd_frame = tk.Frame(parent, bg=CLR["surface"])
            nb_dd_frame.pack(fill="x", pady=(0, 6))
            ttk.Combobox(
                nb_dd_frame, textvariable=self._nb_doc_var,
                values=nb_doc_options, state="readonly", width=70,
                font=("Segoe UI", 9),
            ).pack(side="left")

            self._nb_detail_frame = tk.Frame(parent, bg=CLR["surface"])
            self._nb_detail_frame.pack(fill="x")

            vocab = vectorizer.get_feature_names_out()

            def _render_nb_doc(event=None):
                for w in self._nb_detail_frame.winfo_children():
                    w.destroy()
                p = self._nb_detail_frame

                idx     = nb_doc_options.index(self._nb_doc_var.get())
                x_doc   = X_test[idx]
                if not isinstance(x_doc, _csr):
                    x_doc = _csr(x_doc)

                fname_d = test_filenames[idx] if idx < len(test_filenames) else f"Dokumen Uji #{idx+1}"
                label_d = y_test[idx] if y_test is not None and idx < len(y_test) else "?"
                pred_nb = nb.predict(x_doc)[0]

                # Info dokumen
                hd = tk.Frame(p, bg=CLR["primary_lt"],
                              highlightbackground=CLR["purple"], highlightthickness=1)
                hd.pack(fill="x", pady=(0, 8))
                pred_color = CLR["success"] if str(pred_nb) == str(label_d) else CLR["danger"]
                tk.Label(hd, text=f"📄  {fname_d}",
                         font=("Segoe UI", 9, "bold"), fg=CLR["purple"],
                         bg=CLR["primary_lt"], padx=10, pady=5).pack(side="left")
                tk.Label(hd, text=f"  Label Asli: {label_d}   →   Prediksi NB: {pred_nb}",
                         font=("Segoe UI", 9, "bold"), fg=pred_color,
                         bg=CLR["primary_lt"], padx=10).pack(side="left")

                # Ambil token aktif (TF-IDF > 0) dari dokumen
                x_arr       = x_doc.toarray().flatten()
                active_idx  = np.where(x_arr > 0)[0]
                active_idx  = active_idx[np.argsort(x_arr[active_idx])[::-1]][:15]  # top-15 token

                # Hitung log posterior tiap kelas
                # Gunakan x_arr (dense 1D) agar dot product menghasilkan scalar
                scores = {}
                for c in nb.classes_:
                    lp_c  = float(nb.class_log_prior_[c])
                    flp_c = np.array(nb.feature_log_prob_[c]).flatten()
                    ll    = float(np.dot(x_arr, flp_c))
                    scores[c] = lp_c + ll

                # Tabel skor per kelas
                tk.Label(p, text="Skor Log Posterior per Kelas",
                         font=("Segoe UI", 9, "bold"), fg=CLR["purple"], bg=CLR["surface"]
                         ).pack(anchor="w", pady=(0, 2))

                tbl_sc = tk.Frame(p, bg=CLR["surface"],
                                  highlightbackground=CLR["border"], highlightthickness=1)
                tbl_sc.pack(fill="x", pady=(0, 8))
                hdr_sc = tk.Frame(tbl_sc, bg=CLR["primary_lt"])
                hdr_sc.pack(fill="x")
                for txt, w in [("Kelas", 20), ("log P(C)", 16), ("Σ tfidf×logP(t|C)", 22),
                                ("log P(C|d)", 18), ("Keputusan", 14)]:
                    tk.Label(hdr_sc, text=txt, width=w, font=("Courier New", 12, "bold"),
                             fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                             ).pack(side="left", padx=(6, 0), pady=3)
                tk.Frame(tbl_sc, bg=CLR["border"], height=1).pack(fill="x")

                best_c    = max(scores, key=scores.get)
                for i, c in enumerate(nb.classes_):
                    lp_c  = float(nb.class_log_prior_[c])
                    flp_c = np.array(nb.feature_log_prob_[c]).flatten()
                    ll    = float(np.dot(x_arr, flp_c))
                    score = scores[c]
                    is_w  = (c == best_c)
                    bg    = "#ECFDF5" if is_w else (CLR["surface2"] if i % 2 else CLR["surface"])
                    flag  = "→ PREDIKSI ✓" if is_w else ""
                    row_f = tk.Frame(tbl_sc, bg=bg)
                    row_f.pack(fill="x")
                    for txt, w, clr in [
                        (str(c),        20, CLR["purple"] if is_w else CLR["text_body"]),
                        (f"{lp_c:.4f}", 16, CLR["text_muted"]),
                        (f"{ll:.4f}",   22, CLR["text_muted"]),
                        (f"{score:.4f}",18, CLR["success"] if is_w else CLR["text_body"]),
                        (flag,          14, CLR["success"]),
                    ]:
                        tk.Label(row_f, text=txt, width=w,
                                 font=("Courier New", 12, "bold" if is_w else "normal"),
                                 fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

                # Tabel kontribusi token (top-15 token aktif)
                if len(active_idx) > 0:
                    tk.Label(p, text="Kontribusi Token Aktif (Top-15 TF-IDF) per Kelas",
                             font=("Segoe UI", 9, "bold"), fg=CLR["purple"], bg=CLR["surface"]
                             ).pack(anchor="w", pady=(6, 2))
                    tk.Label(p,
                             text="Kontribusi = tfidf(token) × log P(token | kelas).  "
                                  "Nilai lebih besar → token lebih mendukung kelas tersebut.",
                             font=("Segoe UI", 8, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]
                             ).pack(anchor="w", padx=6)

                    tbl_tok = tk.Frame(p, bg=CLR["surface"],
                                      highlightbackground=CLR["border"], highlightthickness=1)
                    tbl_tok.pack(fill="x", pady=(4, 8))
                    hdr_tok = tk.Frame(tbl_tok, bg=CLR["primary_lt"])
                    hdr_tok.pack(fill="x")
                    col_widths = [22, 10] + [16] * len(nb.classes_)
                    hdrs       = ["Token", "TF-IDF"] + [str(c) for c in nb.classes_]
                    for txt, w in zip(hdrs, col_widths):
                        tk.Label(hdr_tok, text=txt, width=w, font=("Courier New", 12, "bold"),
                                 fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                                 ).pack(side="left", padx=(6, 0), pady=3)
                    tk.Frame(tbl_tok, bg=CLR["border"], height=1).pack(fill="x")

                    for ri, fi in enumerate(active_idx):
                        token    = vocab[fi] if fi < len(vocab) else f"fitur-{fi}"
                        tfidf_v  = float(x_arr[fi])
                        contribs = [
                            tfidf_v * float(np.array(nb.feature_log_prob_[c]).flatten()[fi])
                            for c in nb.classes_
                        ]
                        best_col = int(np.argmax(contribs))

                        bg = CLR["surface2"] if ri % 2 else CLR["surface"]
                        rf = tk.Frame(tbl_tok, bg=bg)
                        rf.pack(fill="x")
                        for col_i, (txt, w) in enumerate(
                            zip([token, f"{tfidf_v:.4f}"] + [f"{v:.4f}" for v in contribs],
                                col_widths)
                        ):
                            is_best_col = (col_i >= 2 and (col_i - 2) == best_col)
                            tk.Label(rf, text=txt, width=w,
                                     font=("Courier New", 12, "bold" if is_best_col else "normal"),
                                     fg=CLR["purple"] if is_best_col else CLR["text_body"],
                                     bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

            # Bind dropdown ke render function
            for w in nb_dd_frame.winfo_children():
                w.bind("<<ComboboxSelected>>", _render_nb_doc)

            _render_nb_doc()

    # ======================================================
    #   VISUALISASI KNN
    # ======================================================

    def _setup_knn_vis_ui(self, card):
        self.knn_vis_card = card
        tk.Label(card, text="Latih model terlebih dahulu untuk melihat visualisasi KNN.",
                 font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=20)

    def _render_knn_vis(self):
        for w in self.knn_vis_card.winfo_children():
            w.destroy()
        parent = self.knn_vis_card
        knn    = self.knn
        ctrl   = self.controller

        # ── Kartu parameter KNN ──────────────────────────────────────
        card_row = tk.Frame(parent, bg=CLR["surface"])
        card_row.pack(fill="x", pady=(0, 8))
        for label, val, color in [
            ("K (tetangga)",  str(self.k_var.get()), CLR["blue"]),
            ("Metrik Jarak",  "Cosine Similarity",   CLR["success"]),
        ]:
            c = tk.Frame(card_row, bg=CLR["primary_lt"],
                         highlightbackground=color, highlightthickness=1)
            c.pack(side="left", padx=(0, 12), pady=4, ipadx=14, ipady=8)
            tk.Label(c, text=label, font=("Segoe UI", 8),
                     fg=CLR["text_muted"], bg=CLR["primary_lt"]).pack()
            tk.Label(c, text=val, font=("Segoe UI", 13, "bold"),
                     fg=color, bg=CLR["primary_lt"]).pack()

        if knn is None:
            tk.Label(parent, text="Latih model terlebih dahulu untuk melihat detail KNN.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=8)
            return

        # ── Kartu Akurasi Latih vs Uji ───────────────────────────────
        knn_train_a = getattr(self, "last_knn_train_acc", None)
        knn_test_a  = getattr(self, "last_knn_acc",       None)
        knn_train_r = getattr(self, "last_knn_train_rep", None)

        if knn_train_a is not None and knn_test_a is not None:
            self._render_train_test_card(
                parent, "KNN", knn_train_a, knn_test_a,
                knn_train_r, getattr(self, "last_knn_report", None),
                CLR["blue"],
            )

        X_test         = getattr(ctrl, "X_test", None)
        test_filenames = getattr(ctrl, "test_filenames", [])
        train_fnames   = getattr(ctrl, "train_filenames", [])
        y_test         = getattr(ctrl, "y_test", None)
        y_train        = getattr(ctrl, "y_train", None)

        if X_test is None or X_test.shape[0] == 0:
            tk.Label(parent, text="Data uji belum tersedia. Jalankan Uji Model terlebih dahulu.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]).pack(pady=8)
            return

        n_test = X_test.shape[0]

        # ── Info ringkas ─────────────────────────────────────────────
        info_f = tk.Frame(parent, bg=CLR["surface"])
        info_f.pack(fill="x", pady=(0, 10))
        for label, value in [
            ("Jumlah Data Latih", str(knn.X_train.shape[0])),
            ("Jumlah Data Uji",   str(n_test)),
            ("Jumlah Fitur TF-IDF", str(knn.X_train.shape[1])),
        ]:
            rf = tk.Frame(info_f, bg=CLR["surface"])
            rf.pack(side="left", padx=(0, 24))
            tk.Label(rf, text=f"{label}:", font=("Segoe UI", 8),
                     fg=CLR["text_muted"], bg=CLR["surface"]).pack(anchor="w")
            tk.Label(rf, text=value, font=("Segoe UI", 11, "bold"),
                     fg=CLR["blue"], bg=CLR["surface"]).pack(anchor="w")

        tk.Frame(parent, bg=CLR["border"], height=1).pack(fill="x", pady=(0, 10))

        # ── Dropdown pilih dokumen uji ───────────────────────────────
        tk.Label(parent,
                 text="Pilih Dokumen Uji untuk Melihat Detail Perhitungan Cosine Similarity",
                 font=("Segoe UI", 10, "bold"), fg=CLR["blue"], bg=CLR["surface"]
                 ).pack(anchor="w", pady=(0, 4))

        dropdown_frame = tk.Frame(parent, bg=CLR["surface"])
        dropdown_frame.pack(fill="x", pady=(0, 8))

        def _doc_label(i):
            fname = test_filenames[i] if i < len(test_filenames) else f"Dokumen Uji #{i+1}"
            label = y_test[i] if y_test is not None and i < len(y_test) else "?"
            return f"[{i+1}] {fname}  (label: {label})"

        doc_options  = [_doc_label(i) for i in range(n_test)]
        self._knn_doc_var = tk.StringVar(value=doc_options[0])

        doc_menu = ttk.Combobox(
            dropdown_frame, textvariable=self._knn_doc_var,
            values=doc_options, state="readonly", width=70,
            font=("Segoe UI", 9),
        )
        doc_menu.pack(side="left")

        # Frame detail — diisi ulang setiap ganti dropdown
        self._knn_detail_frame = tk.Frame(parent, bg=CLR["surface"])
        self._knn_detail_frame.pack(fill="x")

        def _on_doc_select(event=None):
            idx = doc_options.index(self._knn_doc_var.get())
            self._render_knn_detail(idx, knn, X_test, test_filenames,
                                    train_fnames, y_test, y_train)

        doc_menu.bind("<<ComboboxSelected>>", _on_doc_select)

        # Render dokumen pertama langsung
        _on_doc_select()

        # ── Perbandingan metrik jarak (bagian bawah, tetap) ──────────
        tk.Frame(parent, bg=CLR["border"], height=1).pack(fill="x", pady=(16, 0))
        tk.Label(parent, text="Perbandingan Metrik Jarak KNN",
                 font=("Segoe UI", 10, "bold"), fg=CLR["blue"], bg=CLR["surface"]
                 ).pack(anchor="w", pady=(10, 2))

        metric_results = getattr(self, "last_metric_results", None)
        if metric_results:
            formulas = {
                "cosine":    "sim = (a·b) / (‖a‖ ‖b‖)",
                "euclidean": "d   = √Σ(aᵢ−bᵢ)²  → sim=1/(1+d)",
                "manhattan": "d   = Σ|aᵢ−bᵢ|     → sim=1/(1+d)",
            }
            descriptions = {
                "cosine":    "Direkomendasikan untuk TF-IDF sparse",
                "euclidean": "Sensitif terhadap panjang dokumen",
                "manhattan": "Lebih tahan outlier per-dimensi",
            }
            best_metric = max(metric_results, key=lambda r: r["macro_f1"])

            tbl = tk.Frame(parent, bg=CLR["surface"],
                           highlightbackground=CLR["border"], highlightthickness=1)
            tbl.pack(fill="x", pady=(0, 8))
            hdr = tk.Frame(tbl, bg=CLR["primary_lt"])
            hdr.pack(fill="x")
            for txt, w in [("Metrik", 14), ("Formula", 38), ("Macro F1", 12), ("Keterangan", 40), ("Status", 12)]:
                tk.Label(hdr, text=txt, width=w, font=("Courier New", 12, "bold"),
                         fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w").pack(side="left", padx=(6, 0), pady=3)
            tk.Frame(tbl, bg=CLR["border"], height=1).pack(fill="x")

            for i, r in enumerate(metric_results):
                is_best = (r["metric"] == best_metric["metric"])
                bg      = CLR["primary_lt"] if is_best else (CLR["surface2"] if i % 2 else CLR["surface"])
                row_f   = tk.Frame(tbl, bg=bg)
                row_f.pack(fill="x")
                flag = "PILIH ✓" if is_best else ""
                for txt, w in [
                    (r["metric"].capitalize(), 14),
                    (formulas.get(r["metric"], ""), 38),
                    (f"{r['macro_f1']:.4f}", 12),
                    (descriptions.get(r["metric"], ""), 40),
                    (flag, 12),
                ]:
                    tk.Label(row_f, text=txt, width=w,
                             font=("Courier New", 12, "bold" if is_best else "normal"),
                             fg=CLR["primary"] if is_best else CLR["text_body"],
                             bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=3)

            m_labels = [r["metric"].capitalize() for r in metric_results]
            f1s      = [r["macro_f1"] for r in metric_results]
            colors   = [CLR["success"] if r["metric"] == best_metric["metric"] else CLR["blue"]
                        for r in metric_results]
            fig  = Figure(figsize=(6, 2.6), dpi=100)
            ax   = fig.add_subplot(111)
            bars = ax.bar(m_labels, f1s, color=colors, width=0.45)
            ax.bar_label(bars, fmt="%.4f", fontsize=8)
            ax.set_title("Macro F1-Score per Metrik Jarak", fontsize=9, weight="bold")
            ax.set_ylabel("Macro F1-Score", fontsize=8)
            ax.set_ylim(0, min(1.05, max(f1s) * 1.2) if f1s else 1.0)
            ax.set_facecolor(CLR["surface2"])
            fig.tight_layout()
            cv = FigureCanvasTkAgg(fig, master=parent)
            cv.draw()
            cv.get_tk_widget().pack(fill="x", pady=(0, 10))
        else:
            tk.Label(parent,
                     text="Jalankan 'Cari K Terbaik' terlebih dahulu untuk melihat perbandingan metrik jarak.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"], bg=CLR["surface"],
                     wraplength=650).pack(anchor="w", pady=(4, 12))

    def _render_knn_detail(self, doc_idx, knn, X_test, test_filenames,
                           train_fnames, y_test, y_train):
        """
        Render detail perhitungan Cosine Similarity step-by-step
        untuk dokumen uji yang dipilih via dropdown.
        """
        for w in self._knn_detail_frame.winfo_children():
            w.destroy()
        parent = self._knn_detail_frame

        x_q = X_test[doc_idx]
        if not isinstance(x_q, _csr):
            x_q = _csr(x_q)

        fname_q  = test_filenames[doc_idx] if doc_idx < len(test_filenames) else f"Dokumen Uji #{doc_idx+1}"
        label_q  = y_test[doc_idx] if y_test is not None and doc_idx < len(y_test) else "?"
        pred_lbl = knn.predict(x_q)[0]

        # ── Header info dokumen uji ──────────────────────────────────
        hdr_box = tk.Frame(parent, bg=CLR["primary_lt"],
                           highlightbackground=CLR["blue"], highlightthickness=1)
        hdr_box.pack(fill="x", pady=(8, 10))
        tk.Label(hdr_box,
                 text=f"📄  Dokumen Uji: {fname_q}",
                 font=("Segoe UI", 9, "bold"), fg=CLR["blue"],
                 bg=CLR["primary_lt"], padx=10, pady=6).pack(side="left")
        pred_color = CLR["success"] if str(pred_lbl) == str(label_q) else CLR["danger"]
        tk.Label(hdr_box,
                 text=f"  Label Asli: {label_q}   →   Prediksi KNN: {pred_lbl}",
                 font=("Segoe UI", 9, "bold"), fg=pred_color,
                 bg=CLR["primary_lt"], padx=10).pack(side="left")

        # ── Formula cosine similarity ────────────────────────────────
        formula_box = tk.Frame(parent, bg="#F5F3FF",
                               highlightbackground=CLR["purple"], highlightthickness=1)
        formula_box.pack(fill="x", pady=(0, 10))
        tk.Label(formula_box,
                 text="  Formula Cosine Similarity:",
                 font=("Segoe UI", 9, "bold"), fg=CLR["purple"],
                 bg="#F5F3FF", pady=4).pack(anchor="w")
        tk.Label(formula_box,
                 text="         sim(q, d) =   q · d           =      Σ qᵢ × dᵢ\n"
                      "                     ─────────────         ──────────────────────\n"
                      "                      ‖q‖ × ‖d‖            √(Σqᵢ²) × √(Σdᵢ²)",
                 font=("Courier New", 12), fg=CLR["text_body"],
                 bg="#F5F3FF", justify="left", padx=16, pady=2).pack(anchor="w")
        tk.Label(formula_box,
                 text="  q = vektor TF-IDF dokumen uji   |   d = vektor TF-IDF dokumen latih   |   sim ∈ [0, 1]",
                 font=("Segoe UI", 8, "italic"), fg=CLR["text_muted"],
                 bg="#F5F3FF", padx=16, pady=4).pack(anchor="w")

        # ── Hitung cosine similarity ke semua dokumen latih ─────────
        sims      = knn._cosine_similarity(x_q)
        n_train   = knn.X_train.shape[0]

        # Norm dokumen uji
        x_q_arr   = x_q.toarray().flatten()
        norm_q    = float(np.sqrt(np.dot(x_q_arr, x_q_arr)))

        # Top-K index (terurut descending)
        top_k_idx = np.argsort(sims)[-knn.k:][::-1]

        # ── Step 1: Tabel semua similarity ──────────────────────────
        tk.Label(parent,
                 text=f"Langkah 1 — Hitung Cosine Similarity ke Semua {n_train} Dokumen Latih",
                 font=("Segoe UI", 10, "bold"), fg=CLR["blue"], bg=CLR["surface"]
                 ).pack(anchor="w", pady=(4, 2))
        tk.Label(parent,
                 text=f"Norm ‖q‖ dokumen uji = {norm_q:.6f}",
                 font=("Courier New", 12), fg=CLR["text_muted"], bg=CLR["surface"]
                 ).pack(anchor="w", padx=6)

        tbl_all = tk.Frame(parent, bg=CLR["surface"],
                           highlightbackground=CLR["border"], highlightthickness=1)
        tbl_all.pack(fill="x", pady=(4, 8))

        hdr_all = tk.Frame(tbl_all, bg=CLR["primary_lt"])
        hdr_all.pack(fill="x")
        for txt, w in [("No.", 5), ("Nama Dokumen Latih", 36), ("Kelas", 16),
                       ("‖d‖", 10), ("q·d (dot)", 12), ("sim(q,d)", 12), ("Top-K?", 8)]:
            tk.Label(hdr_all, text=txt, width=w, font=("Courier New", 12, "bold"),
                     fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                     ).pack(side="left", padx=(6, 0), pady=3)
        tk.Frame(tbl_all, bg=CLR["border"], height=1).pack(fill="x")

        top_k_set = set(top_k_idx.tolist())

        for i in range(n_train):
            d_arr  = knn.X_train[i].toarray().flatten()
            norm_d = float(knn.X_train_norm[i])
            dot    = float(np.dot(x_q_arr, d_arr))
            sim    = float(sims[i])
            is_top = i in top_k_set

            fname_d  = train_fnames[i] if i < len(train_fnames) else f"dok-{i}"
            label_d  = y_train[i] if y_train is not None and i < len(y_train) else "?"

            bg = "#ECFDF5" if is_top else (CLR["surface2"] if i % 2 else CLR["surface"])
            row_f = tk.Frame(tbl_all, bg=bg)
            row_f.pack(fill="x")

            top_flag = f"✓ K{list(top_k_idx).index(i)+1}" if is_top else ""
            for txt, w, clr in [
                (str(i + 1),        5,  CLR["text_muted"]),
                (fname_d,           36, CLR["blue"] if is_top else CLR["text_body"]),
                (str(label_d),      16, CLR["text_body"]),
                (f"{norm_d:.4f}",   10, CLR["text_muted"]),
                (f"{dot:.4f}",      12, CLR["text_muted"]),
                (f"{sim:.6f}",      12, CLR["success"] if is_top else CLR["text_body"]),
                (top_flag,          8,  CLR["success"]),
            ]:
                tk.Label(row_f, text=txt, width=w, font=("Courier New", 12, "bold" if is_top else "normal"),
                         fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=2)

        # ── Step 2: Top-K detail ─────────────────────────────────────
        tk.Frame(parent, bg=CLR["border"], height=1).pack(fill="x", pady=(6, 6))
        tk.Label(parent,
                 text=f"Langkah 2 — Ambil Top-{knn.k} Tetangga Terdekat",
                 font=("Segoe UI", 10, "bold"), fg=CLR["blue"], bg=CLR["surface"]
                 ).pack(anchor="w", pady=(0, 2))

        tbl_top = tk.Frame(parent, bg=CLR["surface"],
                           highlightbackground=CLR["border"], highlightthickness=1)
        tbl_top.pack(fill="x", pady=(0, 8))
        hdr_top = tk.Frame(tbl_top, bg="#ECFDF5")
        hdr_top.pack(fill="x")
        for txt, w in [("Rank", 6), ("Nama Dokumen Latih", 36),
                       ("Kelas", 16), ("sim(q,d)", 12), ("Bobot Suara", 14)]:
            tk.Label(hdr_top, text=txt, width=w, font=("Courier New", 12, "bold"),
                     fg=CLR["text_hd"], bg="#ECFDF5", anchor="w"
                     ).pack(side="left", padx=(6, 0), pady=3)
        tk.Frame(tbl_top, bg=CLR["border"], height=1).pack(fill="x")

        vote_dict  = {}
        total_sim  = sum(sims[i] for i in top_k_idx)

        for rank, tr_idx in enumerate(top_k_idx, 1):
            fname_d = train_fnames[tr_idx] if tr_idx < len(train_fnames) else f"dok-{tr_idx}"
            label_d = y_train[tr_idx] if y_train is not None and tr_idx < len(y_train) else "?"
            sim_val = float(sims[tr_idx])
            bobot   = sim_val / total_sim if total_sim > 0 else 0.0
            vote_dict[str(label_d)] = vote_dict.get(str(label_d), 0.0) + sim_val

            bg = "#ECFDF5" if rank % 2 else "#F0FFF4"
            row_f = tk.Frame(tbl_top, bg=bg)
            row_f.pack(fill="x")
            for txt, w, clr in [
                (f"#{rank}",        6,  CLR["success"]),
                (fname_d,           36, CLR["blue"]),
                (str(label_d),      16, CLR["text_body"]),
                (f"{sim_val:.6f}",  12, CLR["success"]),
                (f"{bobot:.4f}",    14, CLR["text_muted"]),
            ]:
                tk.Label(row_f, text=txt, width=w, font=("Courier New", 12, "bold"),
                         fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=3)

        # ── Step 3: Voting ───────────────────────────────────────────
        tk.Frame(parent, bg=CLR["border"], height=1).pack(fill="x", pady=(6, 6))
        tk.Label(parent,
                 text="Langkah 3 — Voting Berbobot (Weighted Majority Vote)",
                 font=("Segoe UI", 10, "bold"), fg=CLR["blue"], bg=CLR["surface"]
                 ).pack(anchor="w", pady=(0, 2))
        tk.Label(parent,
                 text="Setiap tetangga memberikan suara ke kelasnya dengan bobot = sim(q,d).\n"
                      "Kelas dengan total bobot terbesar dipilih sebagai prediksi.",
                 font=("Segoe UI", 8, "italic"), fg=CLR["text_muted"], bg=CLR["surface"]
                 ).pack(anchor="w", padx=6)

        tbl_vote = tk.Frame(parent, bg=CLR["surface"],
                            highlightbackground=CLR["border"], highlightthickness=1)
        tbl_vote.pack(fill="x", pady=(4, 8))
        hdr_v = tk.Frame(tbl_vote, bg=CLR["primary_lt"])
        hdr_v.pack(fill="x")
        for txt, w in [("Kelas", 20), ("Total Bobot Suara", 20), ("Proporsi (%)", 16), ("Keputusan", 14)]:
            tk.Label(hdr_v, text=txt, width=w, font=("Courier New", 12, "bold"),
                     fg=CLR["text_hd"], bg=CLR["primary_lt"], anchor="w"
                     ).pack(side="left", padx=(6, 0), pady=3)
        tk.Frame(tbl_vote, bg=CLR["border"], height=1).pack(fill="x")

        winner_cls = max(vote_dict, key=vote_dict.get)
        total_vote = sum(vote_dict.values())

        for i, (cls, bobot) in enumerate(sorted(vote_dict.items(),
                                                 key=lambda x: x[1], reverse=True)):
            is_win = (cls == winner_cls)
            bg     = "#ECFDF5" if is_win else (CLR["surface2"] if i % 2 else CLR["surface"])
            prop   = bobot / total_vote * 100 if total_vote > 0 else 0
            flag   = "→ PREDIKSI ✓" if is_win else ""
            row_f  = tk.Frame(tbl_vote, bg=bg)
            row_f.pack(fill="x")
            for txt, w, clr in [
                (str(cls),        20, CLR["success"] if is_win else CLR["text_body"]),
                (f"{bobot:.6f}",  20, CLR["success"] if is_win else CLR["text_muted"]),
                (f"{prop:.2f}%",  16, CLR["text_muted"]),
                (flag,            14, CLR["success"]),
            ]:
                tk.Label(row_f, text=txt, width=w,
                         font=("Courier New", 12, "bold" if is_win else "normal"),
                         fg=clr, bg=bg, anchor="w").pack(side="left", padx=(6, 0), pady=3)

        # ── Bar chart similarity top-k ───────────────────────────────
        top_labels = []
        for i in top_k_idx:
            fn = train_fnames[i] if i < len(train_fnames) else f"dok-{i}"
            # Potong nama agar tidak terlalu panjang di chart
            top_labels.append(fn[:20] + "…" if len(fn) > 20 else fn)
        top_sims = [float(sims[i]) for i in top_k_idx]

        fig  = Figure(figsize=(7, 2.8), dpi=100)
        ax   = fig.add_subplot(111)
        bars = ax.barh(top_labels[::-1], top_sims[::-1], color=CLR["blue"], height=0.5)
        ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
        ax.set_title(f"Top-{knn.k} Cosine Similarity — {fname_q}", fontsize=9, weight="bold")
        ax.set_xlabel("sim(q, d)", fontsize=8)
        ax.set_xlim(0, min(1.05, max(top_sims) * 1.25) if top_sims else 1.0)
        ax.set_facecolor(CLR["surface2"])
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.draw()
        cv.get_tk_widget().pack(fill="x", pady=(4, 0))

    def _draw_simple_bar_chart(self, parent, labels, values, title, ylabel, color):
        """Menggambar bar chart sederhana di dalam parent frame."""
        fig = Figure(figsize=(7, 2.6), dpi=100)
        ax  = fig.add_subplot(111)
        bars = ax.bar(labels, values, color=color, width=0.55)
        ax.bar_label(bars, fontsize=8)
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_facecolor(CLR["surface2"])
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.draw()
        cv.get_tk_widget().pack(fill="x", padx=20, pady=(0, 8))


    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    @property
    def alpha(self):
        return float("1.0")

    @property
    def k(self):
        return int(str(self.k_var.get()))