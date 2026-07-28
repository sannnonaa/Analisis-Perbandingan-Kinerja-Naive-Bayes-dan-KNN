import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_splitter import (
    SplitConfig, SplitResult, SplitMode, run_split, save_to_controller,
    validate_train_test_labels,
)

CLR = {
    "bg":        "#F0F2F8", "surface":   "#FFFFFF", "surface2":  "#F8FAFF", "primary":   "#4F46E5", "primary_dk":"#3730A3",
    "primary_lt":"#EEF2FF", "success":   "#10B981", "warning":   "#F59E0B", "danger":    "#EF4444", "purple":    "#8B5CF6",
    "blue":      "#3B82F6", "text_hd":   "#1E1B4B", "text_body": "#374151", "text_muted":"#6B7280", "border":    "#E5E7EB",
    "teal":      "#14B8A6", "orange":    "#F97316",
}


class SplitFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=CLR["bg"])
        self.controller = controller

        self._apply_styles()
        self._build_header()

        # Scroll area
        container = tk.Frame(self, bg=CLR["bg"])
        container.pack(fill="both", expand=True)

        self.canvas    = tk.Canvas(container, bg=CLR["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas, bg=CLR["bg"], padx=25, pady=20)

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width)
        )
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        self._build_body(self.scroll_frame)

    # ─────────────────────────────────────────────────────────────────
    # Dipanggil oleh App._show_frame setiap kali tab ini ditampilkan
    # ─────────────────────────────────────────────────────────────────
    def on_tab_shown(self):
        """Refresh kondisi tab saat pengguna berpindah ke tab Pembagian Data."""
        df = getattr(self.controller, "df", None)

        if df is None or "text_clean" not in df.columns:
            self.btn_split.config(state="disabled")
            self.info.config(text="Status: Menunggu data preprocessing...", fg=CLR["warning"])
            return

        self.btn_split.config(state="normal")
        if self.info.cget("text").startswith("Status: Menunggu"):
            self.info.config(text="Status: Siap", fg=CLR["text_muted"])

        # Jika sedang di mode manual, sinkronkan pool train/test dengan
        # data terbaru (mis. setelah preprocessing dijalankan ulang).
        if self.mode_var.get() == "manual":
            self._populate_manual_pool(df)

    def _build_body(self, parent):
        config_card = self._create_card(parent, "Konfigurasi Pembagian Data", CLR["primary"])
        self._setup_config_ui(config_card)

        kpi_card = self._create_card(parent, "Ringkasan Hasil Split", CLR["success"])
        self.kpi_container = tk.Frame(kpi_card, bg=CLR["surface"])
        self.kpi_container.pack(fill="x")

        eda_card = self._create_card(parent, "Visualisasi Distribusi Data", CLR["purple"], expand=True)
        self.eda_container = tk.Frame(eda_card, bg=CLR["surface"])
        self.eda_container.pack(fill="both", expand=True)
        self.eda_container.columnconfigure(0, weight=1)
        self.eda_container.columnconfigure(1, weight=1)

        preview_card = self._create_card(parent, "Preview Data Train & Test", CLR["blue"], expand=True)
        self._build_preview_section(preview_card)

    # ─────────────────────────────────────────────────────────────────
    # Styles
    # ─────────────────────────────────────────────────────────────────
    def _apply_styles(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TProgressbar", thickness=10, troughcolor=CLR["bg"],
                    background=CLR["primary"], borderwidth=0)
        s.configure("Modern.Treeview", background=CLR["surface"],
                    fieldbackground=CLR["surface"], rowheight=26, font=("Segoe UI", 9))
        s.configure("Modern.Treeview.Heading", background=CLR["primary_lt"],
                    foreground=CLR["primary"], font=("Segoe UI", 9, "bold"))
        s.map("Modern.Treeview",
              background=[("selected", CLR["primary_lt"])],
              foreground=[("selected", CLR["primary"])])

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ─────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        outer = tk.Frame(self, bg=CLR["surface"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(side="top", fill="x")
        tk.Frame(outer, bg=CLR["primary"], width=5).pack(side="left", fill="y")
        inner = tk.Frame(outer, bg=CLR["surface"], padx=20, pady=14)
        inner.pack(side="left", fill="x", expand=True)
        tk.Label(
            inner, text="Tahap 4: Pembagian Data Train / Test",
            font=("Segoe UI", 18, "bold"),
            bg=CLR["surface"], fg=CLR["text_hd"]
        ).pack(side="left", anchor="w")

    # ─────────────────────────────────────────────────────────────────
    # Panel Kontrol — skenario A/B DIHAPUS; hanya mode Otomatis / Manual
    # ─────────────────────────────────────────────────────────────────
    def _setup_config_ui(self, parent):
        self.mode_var = tk.StringVar(value="auto")

        tab_row = tk.Frame(parent, bg=CLR["surface"])
        tab_row.pack(fill="x", pady=(0, 10))

        self.btn_tab_auto = tk.Button(
            tab_row, text="Split Otomatis",
            command=lambda: self._switch_mode("auto"),
            bg=CLR["primary"], fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
            padx=14, pady=6,
        )
        self.btn_tab_auto.pack(side="left", padx=(0, 4))

        self.btn_tab_manual = tk.Button(
            tab_row, text="Split Manual (Pilih File Uji)",
            command=lambda: self._switch_mode("manual"),
            bg=CLR["border"], fg=CLR["text_body"],
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            padx=14, pady=6,
        )
        self.btn_tab_manual.pack(side="left")

        tk.Label(
            tab_row,
            text="  Pilih mode: Otomatis (rasio %) atau Manual (tentukan file test sendiri)",
            font=("Segoe UI", 8), bg=CLR["surface"], fg=CLR["text_muted"]
        ).pack(side="left", padx=10)

        # Panel AUTO
        self.panel_auto = tk.Frame(parent, bg=CLR["surface"])
        self.panel_auto.pack(fill="x")
        self._build_auto_panel(self.panel_auto)

        # Panel MANUAL
        self.panel_manual = tk.Frame(parent, bg=CLR["surface"])
        # tidak di-pack dulu; muncul saat tab Manual aktif
        self._build_manual_panel(self.panel_manual)

        # Tombol Jalankan
        btn_row = tk.Frame(parent, bg=CLR["surface"])
        btn_row.pack(fill="x", pady=(10, 0))

        self.btn_split = tk.Button(
            btn_row, text="Lakukan Pembagian Data",
            command=self.run_split,
            bg=CLR["primary"], fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2",
            padx=20, pady=10,
        )
        self.btn_split.pack(side="left")

        self.info = tk.Label(
            btn_row, text="Status: Siap",
            bg=CLR["surface"], fg=CLR["text_muted"], font=("Segoe UI", 9)
        )
        self.info.pack(side="left", padx=20)

        self.progress = ttk.Progressbar(btn_row, length=200, mode="indeterminate",
                                        style="TProgressbar")
        self.progress.pack(side="right")

    # ─────────────────────────────────────────────────────────────────
    # Sub-panel: AUTO
    # ─────────────────────────────────────────────────────────────────
    def _build_auto_panel(self, parent):
        param_row = tk.Frame(parent, bg=CLR["surface"])
        param_row.pack(fill="x", pady=(0, 6))

        tk.Label(param_row, text="Ukuran Test (%):",
                 bg=CLR["surface"], font=("Segoe UI", 9)
                 ).grid(row=0, column=0, padx=(0, 6), sticky="w")

        self.test_size_slider = tk.Scale(
            param_row, from_=10, to=40, orient="horizontal",
            bg=CLR["surface"], length=140, resolution=5,
            activebackground=CLR["primary"],
            command=self._on_slider_change,
        )
        self.test_size_slider.set(20)
        self.test_size_slider.grid(row=0, column=1, padx=(0, 4))

        self.lbl_ratio = tk.Label(
            param_row, text="→  Train 80%  /  Test 20%",
            bg=CLR["surface"], font=("Segoe UI", 9, "bold"), fg=CLR["primary"]
        )
        self.lbl_ratio.grid(row=0, column=2, padx=(0, 20), sticky="w")

        tk.Label(param_row, text="Random State:",
                 bg=CLR["surface"], font=("Segoe UI", 9)
                 ).grid(row=0, column=3, padx=(0, 6), sticky="w")

        self.random_state_var = tk.IntVar(value=42)
        tk.Spinbox(param_row, from_=0, to=999,
                   textvariable=self.random_state_var, width=6,
                   font=("Segoe UI", 9)
                   ).grid(row=0, column=4, padx=(0, 20))

        self.stratify_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            param_row,
            text="Stratify (pertahankan proporsi kelas)",
            variable=self.stratify_var,
            bg=CLR["surface"], font=("Segoe UI", 9),
            activebackground=CLR["surface"],
        ).grid(row=0, column=5, padx=(0, 20))

    # ─────────────────────────────────────────────────────────────────
    # Sub-panel: MANUAL
    # ─────────────────────────────────────────────────────────────────
    def _build_manual_panel(self, parent):
        toolbar = tk.Frame(parent, bg=CLR["surface"])
        toolbar.pack(fill="x", pady=(0, 6))

        tk.Label(toolbar, text="Filter kelas:",
                 bg=CLR["surface"], font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self.filter_class_var = tk.StringVar(value="(Semua)")
        self.filter_class_cb  = ttk.Combobox(
            toolbar, textvariable=self.filter_class_var,
            state="readonly", width=14, font=("Segoe UI", 9)
        )
        self.filter_class_cb.pack(side="left", padx=(0, 12))
        self.filter_class_cb.bind("<<ComboboxSelected>>", self._apply_filter)

        self.lbl_manual_count = tk.Label(
            toolbar, text="Test: 0 file  |  Train: 0 file",
            font=("Segoe UI", 9, "bold"), bg=CLR["surface"], fg=CLR["primary"]
        )
        self.lbl_manual_count.pack(side="right", padx=8)

        cols = tk.Frame(parent, bg=CLR["surface"])
        cols.pack(fill="both", expand=True)

        # Kolom kiri: Train pool
        left_wrap = tk.Frame(cols, bg=CLR["surface"])
        left_wrap.pack(side="left", fill="both", expand=True)
        tk.Label(
            left_wrap, text="Data Train",
            bg=CLR["primary_lt"], fg=CLR["primary"],
            font=("Segoe UI", 9, "bold"), anchor="w", padx=8, pady=5
        ).pack(fill="x")
        self.tree_pool = self._build_manual_tree(left_wrap, CLR["primary"])

        # Tombol tengah
        mid = tk.Frame(cols, bg=CLR["surface"], padx=8)
        mid.pack(side="left", fill="y")
        tk.Frame(mid, bg=CLR["surface"]).pack(expand=True)
        tk.Button(mid, text="»", command=self._move_to_test,
                  bg=CLR["danger"], fg="white",
                  font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
                  padx=10, pady=8).pack(pady=4)
        tk.Button(mid, text="«", command=self._move_to_train,
                  bg=CLR["primary"], fg="white",
                  font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
                  padx=10, pady=8).pack(pady=4)
        tk.Button(mid, text="»»\nSemua", command=self._move_all_to_test,
                  bg="#FEF2F2", fg=CLR["danger"],
                  font=("Segoe UI", 8), relief="flat", cursor="hand2",
                  padx=6, pady=4).pack(pady=(12, 2))
        tk.Button(mid, text="««\nSemua", command=self._move_all_to_train,
                  bg=CLR["primary_lt"], fg=CLR["primary"],
                  font=("Segoe UI", 8), relief="flat", cursor="hand2",
                  padx=6, pady=4).pack(pady=2)
        tk.Frame(mid, bg=CLR["surface"]).pack(expand=True)

        # Kolom kanan: Test set
        right_wrap = tk.Frame(cols, bg=CLR["surface"])
        right_wrap.pack(side="left", fill="both", expand=True)
        tk.Label(
            right_wrap, text="Data Test  (Pilih dari Kolom Train)",
            bg="#FEF2F2", fg=CLR["danger"],
            font=("Segoe UI", 9, "bold"), anchor="w", padx=8, pady=5
        ).pack(fill="x")
        self.tree_test_manual = self._build_manual_tree(right_wrap, CLR["danger"])

        self._pool_data: dict[str, tuple] = {}
        self._test_data: dict[str, tuple] = {}

    def _build_manual_tree(self, parent, accent):
        container = tk.Frame(parent, bg=CLR["surface"])
        container.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            container, columns=("filename", "label"),
            show="headings", height=12,
            style="Modern.Treeview", selectmode="extended"
        )
        tree.heading("filename", text="Nama File")
        tree.heading("label",    text="Label / Kelas")
        tree.column("filename", width=260)
        tree.column("label",    width=110, anchor="center")
        tree.tag_configure(accent, foreground=accent)
        sb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    # ─────────────────────────────────────────────────────────────────
    # Logika pemindahan baris (manual mode)
    # ─────────────────────────────────────────────────────────────────
    def _move_to_test(self):
        for iid in self.tree_pool.selection():
            vals = self.tree_pool.item(iid, "values")
            fn   = vals[0]
            self.tree_pool.delete(iid)
            self._pool_data.pop(fn, None)
            if fn not in self._test_data:
                self._test_data[fn] = vals
                self.tree_test_manual.insert("", "end", iid=fn, values=vals)
        self._update_manual_count()

    def _move_to_train(self):
        for iid in self.tree_test_manual.selection():
            vals = self.tree_test_manual.item(iid, "values")
            fn   = vals[0]
            self.tree_test_manual.delete(iid)
            self._test_data.pop(fn, None)
            if fn not in self._pool_data:
                self._pool_data[fn] = vals
                self.tree_pool.insert("", "end", iid=fn, values=vals)
        self._update_manual_count()

    def _move_all_to_test(self):
        for iid in self.tree_pool.get_children():
            vals = self.tree_pool.item(iid, "values")
            fn   = vals[0]
            self.tree_pool.delete(iid)
            self._pool_data.pop(fn, None)
            if fn not in self._test_data:
                self._test_data[fn] = vals
                self.tree_test_manual.insert("", "end", iid=fn, values=vals)
        self._update_manual_count()

    def _move_all_to_train(self):
        for iid in self.tree_test_manual.get_children():
            vals = self.tree_test_manual.item(iid, "values")
            fn   = vals[0]
            self.tree_test_manual.delete(iid)
            self._test_data.pop(fn, None)
            if fn not in self._pool_data:
                self._pool_data[fn] = vals
                self.tree_pool.insert("", "end", iid=fn, values=vals)
        self._update_manual_count()

    def _update_manual_count(self):
        n_test  = len(self._test_data)
        n_train = len(self._pool_data)
        self.lbl_manual_count.config(
            text=f"Test: {n_test} file  |  Train: {n_train} file",
            fg=CLR["danger"] if n_test == 0 else CLR["success"]
        )

    def _apply_filter(self, *_):
        kelas = self.filter_class_var.get()
        for iid in self.tree_pool.get_children():
            self.tree_pool.delete(iid)
        for fn, vals in self._pool_data.items():
            _, lbl_val = vals
            if kelas == "(Semua)" or lbl_val == kelas:
                self.tree_pool.insert("", "end", iid=fn, values=vals)

    def _populate_manual_pool(self, df):
        for iid in self.tree_pool.get_children():
            self.tree_pool.delete(iid)
        self._pool_data.clear()

        current_fns = set(df["filename"].tolist())
        for fn in list(self._test_data.keys()):
            if fn not in current_fns:
                del self._test_data[fn]
                try:
                    self.tree_test_manual.delete(fn)
                except tk.TclError:
                    pass

        for _, row in df.iterrows():
            fn  = row["filename"]
            lbl = row["label"]
            if fn not in self._test_data:
                self._pool_data[fn] = (fn, lbl)
                self.tree_pool.insert("", "end", iid=fn, values=(fn, lbl))

        classes = ["(Semua)"] + sorted(df["label"].unique().tolist())
        self.filter_class_cb["values"] = classes
        self.filter_class_var.set("(Semua)")
        self._update_manual_count()

    # ─────────────────────────────────────────────────────────────────
    # Tab switch
    # ─────────────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        self.mode_var.set(mode)
        if mode == "auto":
            self.panel_manual.pack_forget()
            self.panel_auto.pack(fill="x")
            self.btn_tab_auto.config(bg=CLR["primary"], fg="white",
                                     font=("Segoe UI", 9, "bold"))
            self.btn_tab_manual.config(bg=CLR["border"], fg=CLR["text_body"],
                                       font=("Segoe UI", 9))
        else:
            self.panel_auto.pack_forget()
            self.panel_manual.pack(fill="both", expand=True)
            self.btn_tab_manual.config(bg=CLR["danger"], fg="white",
                                       font=("Segoe UI", 9, "bold"))
            self.btn_tab_auto.config(bg=CLR["border"], fg=CLR["text_body"],
                                     font=("Segoe UI", 9))
            df = getattr(self.controller, "df", None)
            if df is not None and "text_clean" in df.columns:
                self._populate_manual_pool(df)

    def _on_slider_change(self, val):
        ts = int(val)
        self.lbl_ratio.config(text=f"→  Train {100 - ts}%  /  Test {ts}%")

    # ─────────────────────────────────────────────────────────────────
    # Card helpers
    # ─────────────────────────────────────────────────────────────────
    def _create_card(self, parent, title, accent_color, expand=False):
        outer = tk.Frame(parent, bg=CLR["surface"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        outer.pack(fill="both" if expand else "x", expand=expand, pady=(0, 20))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=15, pady=10)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 11, "bold"),
                 bg=CLR["primary_lt"], fg=accent_color).pack(side="left")
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")
        content = tk.Frame(outer, bg=CLR["surface"], padx=15, pady=15)
        content.pack(fill="both", expand=True)
        return content

    def _make_eda_card(self, title, accent, row, col, padx=(0, 0)):
        outer = tk.Frame(self.eda_container, bg=CLR["surface"],
                         highlightbackground=CLR["border"], highlightthickness=1)
        outer.grid(row=row, column=col, sticky="nsew", padx=padx, pady=(0, 14))
        hd = tk.Frame(outer, bg=CLR["primary_lt"], padx=12, pady=7)
        hd.pack(fill="x")
        tk.Label(hd, text=title, font=("Segoe UI", 10, "bold"),
                 bg=CLR["primary_lt"], fg=accent).pack(side="left")
        tk.Frame(outer, bg=accent, height=3).pack(fill="x")
        body = tk.Frame(outer, bg=CLR["surface"], padx=8, pady=8)
        body.pack(fill="both", expand=True)
        return body

    def _build_preview_section(self, parent):
        preview_row = tk.Frame(parent, bg=CLR["surface"])
        preview_row.pack(fill="x")
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=1)

        left = tk.Frame(preview_row, bg=CLR["surface"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(left, text="Data Train",
                 bg=CLR["primary_lt"], fg=CLR["primary"],
                 font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        self.tree_train = self._build_preview_tree(left)

        right = tk.Frame(preview_row, bg=CLR["surface"])
        right.grid(row=0, column=1, sticky="nsew")
        tk.Label(right, text="Data Test",
                 bg="#FEF2F2", fg=CLR["danger"],
                 font=("Segoe UI", 9, "bold"), anchor="w", padx=10, pady=6).pack(fill="x")
        self.tree_test = self._build_preview_tree(right)

    def _build_preview_tree(self, parent):
        container = tk.Frame(parent, bg=CLR["surface"])
        container.pack(fill="both", expand=True)
        tree = ttk.Treeview(
            container, columns=("no", "filename", "label"),
            show="headings", height=10, style="Modern.Treeview"
        )
        tree.heading("no",       text="#")
        tree.heading("filename", text="Nama File")
        tree.heading("label",    text="Label")
        tree.column("no",       width=40,  anchor="center")
        tree.column("filename", width=280)
        tree.column("label",    width=100, anchor="center")
        sb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return tree

    # ─────────────────────────────────────────────────────────────────
    # Logika split — skenario A/B dihapus, selalu pakai df penuh
    # ─────────────────────────────────────────────────────────────────
    def run_split(self):
        df = self.controller.df
        if df is None or "text_clean" not in df.columns:
            messagebox.showwarning("Peringatan", "Lakukan preprocessing terlebih dahulu.")
            return

        mode = self.mode_var.get()

        if mode == "manual":
            if not self._pool_data and not self._test_data:
                self._populate_manual_pool(df)
            if not self._test_data:
                messagebox.showwarning(
                    "Mode Manual",
                    "Belum ada file yang dipilih sebagai Test Set.\n"
                    "Pindahkan minimal 1 file ke kolom Test (kanan) terlebih dahulu."
                )
                return
            config = SplitConfig(
                mode=SplitMode.MANUAL,
                manual_test_filenames=set(self._test_data.keys()),
            )
            mode_label = "Mode Manual (file dipilih sendiri)"
        else:
            config = SplitConfig(
                mode=SplitMode.AUTO,
                test_size    = self.test_size_slider.get() / 100.0,
                random_state = self.random_state_var.get(),
                stratify     = self.stratify_var.get(),
            )
            mode_label = "Mode Otomatis (stratified split)"

        self.btn_split.config(state="disabled")
        self.progress.start(10)
        self.info.config(text=f"Status: Memproses... [{mode_label}]", fg=CLR["warning"])
        threading.Thread(
            target=self._worker, args=(df, config, mode_label), daemon=True
        ).start()

    def _worker(self, df, config: SplitConfig, mode_label: str = ""):
        try:
            result = run_split(df, config)
            self.after(0, lambda: self._finish(df, result, mode_label))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self._on_error(msg))

    def _finish(self, df, result: SplitResult, mode_label: str = ""):
        missing = validate_train_test_labels(result.y_train, result.y_test)
        if missing:
            self.progress.stop()
            self.btn_split.config(state="normal")
            msg = (
                "Test set mengandung kelas yang tidak muncul di train set:\n"
                + ", ".join(sorted(map(str, missing)))
                + "\n\nModel tidak dapat mempelajari kelas tersebut. "
                "Pilih ulang pembagian data atau tambahkan data train untuk kelas tersebut."
            )
            messagebox.showerror("Validasi Split Gagal", msg)
            return

        save_to_controller(self.controller, result)

        # Simpan df yang dipakai split ke controller agar K-Fold CV
        # membaca subset yang benar (bukan selalu df lengkap).
        self.controller.df_split  = df
        self.controller.split_mode = self.mode_var.get()   # "auto" | "manual"

        self._render_kpis(result)
        self._render_eda_charts(result)
        self._render_preview(df, result)

        self.progress.stop()
        self.btn_split.config(state="normal")
        self.info.config(text=f"Status: Selesai [{mode_label}]", fg=CLR["success"])

        messagebox.showinfo(
            "Sukses",
            f"Data berhasil dibagi  [{mode_label}].\n"
            f"Lanjutkan ke tahap Ekstraksi Fitur TF-IDF."
        )

    def _on_error(self, msg):
        messagebox.showerror("Error", msg)
        self.btn_split.config(state="normal")
        self.progress.stop()
        self.info.config(text="Status: Error", fg=CLR["danger"])

    # ─────────────────────────────────────────────────────────────────
    # Render KPI
    # ─────────────────────────────────────────────────────────────────
    def _render_kpis(self, result: SplitResult):
        for w in self.kpi_container.winfo_children():
            w.destroy()
        kpis = [
            ("Data Train",
             f"{result.n_train}  ({result.ratio_train:.0f}%)",
             CLR["primary"], CLR["primary_lt"]),
            ("Data Test",
             f"{result.n_test}  ({result.ratio_test:.0f}%)",
             CLR["danger"], "#FEF2F2"),
        ]
        for i, (title, value, fg, bg) in enumerate(kpis):
            kf = tk.Frame(self.kpi_container, bg=bg,
                          highlightbackground=fg, highlightthickness=2)
            kf.pack(side="left", fill="both", expand=True,
                    padx=(0, 8 if i < len(kpis) - 1 else 0))
            tk.Frame(kf, bg=fg, height=4).pack(fill="x")
            tk.Label(kf, text=value, font=("Segoe UI", 16, "bold"),
                     bg=bg, fg=fg).pack(pady=(10, 2))
            tk.Label(kf, text=title, font=("Segoe UI", 8),
                     bg=bg, fg=fg, wraplength=110).pack(pady=(0, 10))

    # ─────────────────────────────────────────────────────────────────
    # Render EDA chart
    # ─────────────────────────────────────────────────────────────────
    def _render_eda_charts(self, result: SplitResult):
        for w in self.eda_container.winfo_children():
            w.destroy()
        self._chart_distribusi_kelas(result)
        self._chart_source_type(result)
        self.scroll_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _chart_distribusi_kelas(self, result: SplitResult):
        c = self._make_eda_card(
            "Distribusi Kelas: Train vs Test", CLR["primary"],
            row=0, col=0, padx=(0, 8)
        )
        fig = Figure(figsize=(5.5, 3.8), dpi=90)
        fig.patch.set_facecolor(CLR["surface"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(CLR["surface2"])

        classes    = list(result.class_dist.keys())
        train_vals = [result.class_dist[k]["train"] for k in classes]
        test_vals  = [result.class_dist[k]["test"]  for k in classes]
        x = np.arange(len(classes))
        w = 0.38

        ax.bar(x - w/2, train_vals, width=w, label="Train",
               color=CLR["primary"], alpha=0.85, edgecolor="white", zorder=3)
        ax.bar(x + w/2, test_vals,  width=w, label="Test",
               color=CLR["danger"],  alpha=0.85, edgecolor="white", zorder=3)
        ax.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)

        for xi, (tv, tsv) in enumerate(zip(train_vals, test_vals)):
            ax.text(xi - w/2, tv + 0.15, str(tv), ha="center", fontsize=7,
                    color=CLR["primary"], fontweight="bold")
            color = "#DC2626" if tsv <= 1 else CLR["danger"]
            ax.text(xi + w/2, tsv + 0.15, f"{tsv}{'⚠' if tsv <= 1 else ''}",
                    ha="center", fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Jumlah Dokumen", fontsize=8, color=CLR["text_muted"])
        ax.set_title(
            "Test ≤ 1 dokumen maka metrik tidak representatif\n(tanda ⚠ pada nilai)",
            fontsize=8, color=CLR["text_muted"], pad=5
        )
        ax.legend(fontsize=8, framealpha=0.7)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="both", expand=True)

    def _chart_source_type(self, result: SplitResult):
        c = self._make_eda_card(
            "Komposisi Data: Riil vs Sintetis", CLR["teal"],
            row=0, col=1, padx=(8, 0)
        )

        df = getattr(self.controller, "df", None)
        if df is None or "source_type" not in df.columns:
            tk.Label(c, text="Kolom source_type tidak tersedia.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"],
                     bg=CLR["surface"]).pack(pady=20)
            return

        src     = df["source_type"].fillna("unknown").values
        tr_src  = src[result.train_idx]
        ts_src  = src[result.test_idx]

        labels_src   = ["real", "synthetic", "unknown"]
        train_counts = [int((tr_src == s).sum()) for s in labels_src]
        test_counts  = [int((ts_src == s).sum()) for s in labels_src]

        pairs = [
            (l, tr, ts)
            for l, tr, ts in zip(labels_src, train_counts, test_counts)
            if tr > 0 or ts > 0
        ]
        if not pairs:
            tk.Label(c, text="Tidak ada data source_type.",
                     font=("Segoe UI", 9, "italic"), fg=CLR["text_muted"],
                     bg=CLR["surface"]).pack(pady=20)
            return

        labels_f, tr_f, ts_f = zip(*pairs)
        x = np.arange(len(labels_f))
        w = 0.38

        fig = Figure(figsize=(5.5, 3.8), dpi=90)
        fig.patch.set_facecolor(CLR["surface"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(CLR["surface2"])

        ax.bar(x - w/2, tr_f, width=w, label="Train",
               color=CLR["teal"],   alpha=0.85, edgecolor="white", zorder=3)
        ax.bar(x + w/2, ts_f, width=w, label="Test",
               color=CLR["orange"], alpha=0.85, edgecolor="white", zorder=3)
        ax.grid(axis="y", color=CLR["border"], linestyle="--", linewidth=0.5, zorder=0)

        for xi, (tv, tsv) in enumerate(zip(tr_f, ts_f)):
            ax.text(xi - w/2, tv + 0.1, str(tv), ha="center", fontsize=7,
                    color=CLR["teal"],   fontweight="bold")
            ax.text(xi + w/2, tsv + 0.1, str(tsv), ha="center", fontsize=7,
                    color=CLR["orange"], fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_f, fontsize=9)
        ax.set_ylabel("Jumlah Dokumen", fontsize=8, color=CLR["text_muted"])
        ax.set_title("Distribusi source_type per Split", fontsize=8, color=CLR["text_muted"])
        ax.legend(fontsize=8, framealpha=0.7)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=c).get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────
    # Render tabel preview
    # ─────────────────────────────────────────────────────────────────
    def _render_preview(self, df, result: SplitResult):
        for tree, idx in [
            (self.tree_train, result.train_idx),
            (self.tree_test,  result.test_idx),
        ]:
            for item in tree.get_children():
                tree.delete(item)
            for rank, df_i in enumerate(idx, 1):
                row = df.iloc[df_i]
                tree.insert("", "end", values=(rank, row["filename"], row["label"]))