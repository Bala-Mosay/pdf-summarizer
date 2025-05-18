# 📄 PDF Summarizer

A powerful and flexible tool for summarizing large PDF documents. This project extracts content from PDFs—including scanned documents with images—and generates concise summaries using advanced NLP techniques. It is particularly useful for researchers, students, and professionals needing fast, accurate overviews of academic papers, reports, or lengthy articles.

---

## ✨ Features

- ✅ Multi-page PDF summarization
- 🧠 NLP-based summarization using Transformer models (e.g., BERT, T5)
- 🖼️ Supports OCR for image-based PDFs (Tesseract)
- 📃 Clean and structured summary output
- 📈 Evaluation support using **ROUGE** metrics
- 🎓 Ideal for academic research and literature review workflows

---

## 🛠️ Technologies Used

- Python
- [PyMuPDF](https://pymupdf.readthedocs.io/) / [pdfplumber](https://github.com/jsvine/pdfplumber)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) *(optional for image-based PDFs)*
- [ROUGE](https://pypi.org/project/rouge/) for summary evaluation

---

## 🚀 Getting Started

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/pdf-summarizer.git
cd pdf-summarizer
