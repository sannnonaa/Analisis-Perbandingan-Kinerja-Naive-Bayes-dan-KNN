import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ocr_extraction import extract_text
from preprocessing import preprocess_text

class SingleFileFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#F1F5F9")
        self.controller = controller
        self.image_ref = None
        self.canvas_tfidf = None

        #header
        header = tk.Frame(self, bg="#FFFFFF", height=70, highlightbackground="#E2E8F0", highlightthickness=1)
        header.pack(side="top", fill="x")
        tk.Label(header, text="Uji Real-time Dokumen Tunggal", 
                font=("Poppins", 14, "bold"), bg="#FFFFFF", fg="#1E1B4B").pack(side="left", padx=30, pady=15)

        #main container
        main_container = tk.Frame(self, bg="#F1F5F9", padx=25, pady=20)
        main_container.pack(fill="both", expand=True)

        #preview gambar dokumen
        self.left_col = tk.Frame(main_container, bg="#F1F5F9")
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 20))

        tk.Label(self.left_col, text="Pratinjau Visual Dokumen", font=("Poppins", 10, "bold"), bg="#F1F5F9", fg="#475569").pack(anchor="w", pady=(0, 10))
        
        preview_border = tk.Frame(self.left_col, bg="#FFFFFF", highlightbackground="#E2E8F0", highlightthickness=1)
        preview_border.pack(fill="both", expand=True)
        
        self.canvas_pdf = tk.Canvas(preview_border, bg="#F8FAFC", bd=0, highlightthickness=0)
        self.canvas_pdf.pack(fill="both", expand=True)

        #hasil klasifikasi dan fitur tf-idf
        self.right_col = tk.Frame(main_container, bg="#F1F5F9", width=450)
        self.right_col.pack(side="right", fill="both")
        self.right_col.pack_propagate(False)

        #card upload
        upload_card = tk.Frame(self.right_col, bg="#FFFFFF", padx=20, pady=25, highlightbackground="#E2E8F0", highlightthickness=1)
        upload_card.pack(fill="x", pady=(0, 15))

        tk.Button(upload_card, text="Pilih File", command=self.pick_file,
                bg="#4F46E5", fg="white", font=("Poppins", 10, "bold"), 
                relief="flat", cursor="hand2", padx=25, pady=12).pack(fill="x")
        
        self.file_info = tk.Label(upload_card, text="Format: .pdf, .docx", font=("Inter", 8, "italic"), bg="#FFFFFF", fg="#94A3B8")
        self.file_info.pack(pady=(8, 0))
        
        self.progress = ttk.Progressbar(upload_card, mode="indeterminate")
        self.progress.pack(fill="x", pady=(15, 0))

        #card hasil klasifikasi
        self.res_card = tk.LabelFrame(self.right_col, text=" Hasil Prediksi ", font=("Poppins", 9, "bold"), 
                                    bg="#FFFFFF", fg="#4F45E5", padx=15, pady=15)
        self.res_card.pack(fill="x", pady=(0, 15))
        
        self.status_label = tk.Label(self.res_card, text="Menunggu input...", bg="#FFFFFF", fg="#94A3B8", font=("Helvetica", 9, "italic"))
        self.status_label.pack(pady=20)

        #card fitur tf-idf
        self.keyword_card = tk.LabelFrame(self.right_col, text=" Kata Kunci Terpenting (TF-IDF) ", font=("Poppins", 9, "bold"), 
                                        bg="#FFFFFF", fg="#4F46E5", padx=15, pady=15)
        self.keyword_card.pack(fill="both", expand=True)
        
        self.chart_area = tk.Frame(self.keyword_card, bg="#FFFFFF")
        self.chart_area.pack(fill="both", expand=True)

    def render_pdf_preview(self, path):
        try:
            self.update_idletasks()
            w, h = self.canvas_pdf.winfo_width(), self.canvas_pdf.winfo_height()
            doc = fitz.open(path)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
            self.image_ref = ImageTk.PhotoImage(img)
            self.canvas_pdf.delete("all")
            self.canvas_pdf.create_image(w//2, h//2, anchor="center", image=self.image_ref)
        except:
            pass

    def pick_file(self):
        if not hasattr(self.controller, 'vectorizer') or self.controller.vectorizer is None:
            messagebox.showwarning("Peringatan", "Model/TF-IDF belum siap. Selesaikan tahap Analysis dahulu.")
            return

        path = filedialog.askopenfilename(filetypes=[("Documents", "*.pdf *.docx"), ("All", "*.*")])
        if not path: return

        self.file_info.config(text=f"File: {path.split('/')[-1]}")
        for w in self.res_card.winfo_children(): w.destroy()
        self.progress.start(10)

        if path.lower().endswith('.pdf'): self.render_pdf_preview(path)

        def worker():
            try:
                # Bug #1 fix: unpack tuple (text, metode)
                # Bug #2 fix: gunakan preprocess_text() agar konsisten dengan pipeline utama
                raw_text, _ = extract_text(path, lang="ind")
                clean_text = preprocess_text(raw_text)

                #bangun tf-idf
                X_vec = self.controller.vectorizer.transform([clean_text])
                feature_names = self.controller.vectorizer.get_feature_names_out()

                # Bug #3 fix: ambil baris pertama secara eksplisit agar .data
                # hanya berisi elemen baris tersebut, bukan seluruh sparse matrix
                row = X_vec[0]
                nz_idx = row.nonzero()[1]
                weights = sorted(zip(feature_names[nz_idx], row.data), key=lambda x: x[1], reverse=True)[:10]

                #klasifikasi
                nb_p = self.controller.nb_model.predict(X_vec)[0]
                knn_p = self.controller.knn_model.predict(X_vec)[0]

                def update_ui():
                    self.progress.stop()
                    self.create_result_badge("Naive Bayes", nb_p, "#EFF6FF", "#2563EB")
                    self.create_result_badge("KNN", knn_p, "#F0FDF4", "#16A34A")

                    self.draw_keyword_chart(weights)

                self.after(0, update_ui)
            except Exception as e:
                self.after(0, lambda: [self.progress.stop(), messagebox.showerror("Error", str(e))])

        threading.Thread(target=worker, daemon=True).start()

    def create_result_badge(self, model, pred, bg_color, fg_color):
        f = tk.Frame(self.res_card, bg=bg_color, pady=8, padx=15, highlightbackground=fg_color, highlightthickness=1)
        f.pack(fill="x", pady=4)
        tk.Label(f, text=f"{model}:", font=("Poppins", 8), bg=bg_color, fg=fg_color).pack(side="left")
        tk.Label(f, text=pred.upper(), font=("Poppins", 11, "bold"), bg=bg_color, fg=fg_color).pack(side="right")

    def draw_keyword_chart(self, top_keywords):
        if self.canvas_tfidf: self.canvas_tfidf.get_tk_widget().destroy()
        if not top_keywords: return

        words, scores = zip(*top_keywords[::-1])

        fig = Figure(figsize=(4, 3), dpi=90)
        ax = fig.add_subplot(111)
        ax.barh(words, scores, color="#3B82F6")
        ax.set_title("Bobot Kata Terkuat", fontsize=9, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=8)

        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        
        fig.tight_layout()
        self.canvas_tfidf = FigureCanvasTkAgg(fig, master=self.chart_area)
        self.canvas_tfidf.draw()
        self.canvas_tfidf.get_tk_widget().pack(fill="both", expand=True)