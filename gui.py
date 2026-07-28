import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import os

from GUI.business_frame import BusinessFrame
from GUI.data_frame import DataFramePage
from GUI.ocr_frame import OCRFrame
from GUI.preprocess_frame import PreprocessFrame
from GUI.split_frame import SplitFrame
from GUI.tfidf_frame import TfidfFrame
from GUI.modeling_frame import ModelingFrame
from GUI.analysis_frame import AnalysisFrame
from GUI.file_frame import SingleFileFrame


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Sistem Klasifikasi Dokumen - PT Atika Jaya Samudera")
        self.tk.call('tk', 'scaling', 1.75)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.test_filenames = None

        self.df = None
        self.dataset_path = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.nb_model = None
        self.knn_model = None

        self.train_idx = None
        self.test_idx  = None

        self.X_real = None
        self.y_real = None
        self.X_real_train = None
        self.y_real_train = None
        self.X_real_test  = None
        self.y_real_test  = None

        self.X_full_train = None
        self.y_full_train = None

        self.full_train_texts  = []
        self.full_train_labels = None
        self.full_test_texts   = []
        self.full_test_labels  = None
        self.real_train_texts  = []
        self.real_train_labels = None
        self.real_test_texts   = []
        self.real_test_labels  = None

        # Main Wrapper
        self.wrapper = tk.Frame(self, bg="#F1F5F9")
        self.wrapper.pack(fill="both", expand=True)

        self._create_sidebar()
        self._create_main_area()

        # Inisialisasi Frame
        self.frames = {}

        for F in (
            BusinessFrame,
            DataFramePage,
            OCRFrame,
            PreprocessFrame,
            SplitFrame,
            TfidfFrame,
            SingleFileFrame
        ):
            frame = F(self.content, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Simpan referensi split_frame agar preprocess_frame bisa notify
        self.split_frame = self.frames[SplitFrame]

        modeling = ModelingFrame(self.content,self)
        self.frames[ModelingFrame] = modeling
        modeling.grid(row=0, column=0, sticky="nsew")

        analysis = AnalysisFrame(self.content, self, modeling_frame=modeling)
        self.frames[AnalysisFrame] = analysis
        analysis.grid(row=0, column=0, sticky="nsew")

        modeling.on_results = analysis.update_from_test

        self._show_frame(BusinessFrame)

    def _on_closing(self):
        """Membunuh semua proses dan terminal saat aplikasi di-close"""
        self.quit()
        self.destroy()
        os._exit(0)

    # SIDEBAR
    def _create_sidebar(self):
        self.sidebar = tk.Frame(self.wrapper, width=240, bg="#1E293B")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Title
        title = tk.Label(
            self.sidebar,
            text="📄 DOCUMENTS\nCLASSIFICATION",
            bg="#1E293B",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            justify="center"
        )
        title.pack(pady=(40, 30), padx=20)

        # Separator
        separator = tk.Frame(self.sidebar, height=2, bg="#334155")
        separator.pack(fill="x", padx=20, pady=(0, 20))

        self.menu_buttons = {}
        self.active_indicator = None

        menus = [
            ("🏠  Dashboard", BusinessFrame),
            ("📂  Akuisisi Data", DataFramePage),
            ("🔍  Ekstraksi Teks", OCRFrame),
            ("⚙️  Pra-pemrosesan", PreprocessFrame),
            ("🔀  Pembagian Data", SplitFrame),
            ("📊  Ekstraksi Fitur", TfidfFrame),
            ("🤖  Permodelan", ModelingFrame),
            ("📈  Analisis", AnalysisFrame)
        ]

        for text, frame_cls in menus:
            btn_frame = tk.Frame(self.sidebar, bg="#1E293B")
            btn_frame.pack(fill="x", padx=10, pady=2)

            # Indicator bar (hidden by default)
            indicator = tk.Frame(btn_frame, bg="#1E293B", width=4)
            indicator.pack(side="left", fill="y")

            btn = tk.Button(
                btn_frame,
                text=text,
                bg="#1E293B",
                fg="#CBD5E1",
                bd=0,
                anchor="w",
                padx=15,
                pady=12,
                font=("Segoe UI", 14),
                cursor="hand2",
                activebackground="#334155",
                activeforeground="white",
                command=lambda f=frame_cls: self._show_frame(f)
            )
            btn.pack(side="left", fill="x", expand=True)

            # Store references
            self.menu_buttons[frame_cls] = {
                "button": btn,
                "indicator": indicator,
                "frame": btn_frame
            }

            # Hover Effects
            btn.bind("<Enter>", lambda e, f=frame_cls: self._on_hover(f))
            btn.bind("<Leave>", lambda e, f=frame_cls: self._off_hover(f))

    def _on_hover(self, frame_cls):
        btn_data = self.menu_buttons[frame_cls]
        if btn_data["button"]["fg"] != "white":
            btn_data["frame"].config(bg="#334155")
            btn_data["button"].config(bg="#334155")
            btn_data["indicator"].config(bg="#334155")

    def _off_hover(self, frame_cls):
        btn_data = self.menu_buttons[frame_cls]
        if btn_data["button"]["fg"] != "white":
            btn_data["frame"].config(bg="#1E293B")
            btn_data["button"].config(bg="#1E293B")
            btn_data["indicator"].config(bg="#1E293B")

    # MAIN AREA
    def _create_main_area(self):
        self.main = tk.Frame(self.wrapper, bg="#F1F5F9")
        self.main.pack(side="right", fill="both", expand=True)

        # Content Area
        self.content = tk.Frame(self.main, bg="#F1F5F9")
        self.content.pack(fill="both", expand=True, padx=20, pady=20)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

    # FRAME NAVIGATION
    def _show_frame(self, cls):
        frame = self.frames[cls]
        frame.tkraise()

        # Reset semua button sidebar
        for f, btn_data in self.menu_buttons.items():
            btn_data["button"].config(
                bg="#1E293B",
                fg="#CBD5E1",
                font=("Segoe UI", 10)
            )
            btn_data["frame"].config(bg="#1E293B")
            btn_data["indicator"].config(bg="#1E293B")

        # Highlight active button
        if cls in self.menu_buttons:
            current_data = self.menu_buttons[cls]
            current_data["button"].config(
                bg="#334155",
                fg="white",
                font=("Segoe UI", 10, "bold")
            )
            current_data["frame"].config(bg="#334155")
            current_data["indicator"].config(bg="#38BDF8")  # Blue indicator

        # Logic khusus saat pindah frame
        if cls == PreprocessFrame:
            frame.refresh_doc_list()

        if cls == SplitFrame:
            frame.on_tab_shown()


    # PUBLIC METHOD (untuk dipanggil dari frame lain)
    def show_frame(self, cls):
        """Public method untuk navigasi dari frame lain"""
        self._show_frame(cls)


if __name__ == "__main__":
    app = App()
    app.mainloop()