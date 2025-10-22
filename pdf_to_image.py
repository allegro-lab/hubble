import os
import fitz  # PyMuPDF
from PIL import Image

def pdf_to_images_pymupdf(pdf_path, output_folder, zoom=4, fmt="png"):
    """
    Convert a PDF to high-resolution images using PyMuPDF.
    Args:
        pdf_path: path to the PDF
        output_folder: where to save images
        zoom: scale factor (2=~150dpi, 4=~300dpi, 8=~600dpi)
        fmt: 'png' or 'jpeg'
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"\nConverting '{pdf_name}' with {len(doc)} pages...")
    saved = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(zoom, zoom)  # control resolution
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.{fmt}")
        pix.save(out_path)
        saved.append(out_path)
        print(f"[OK] {out_path}")
    doc.close()
    return saved


def convert_all_pdfs(folder_path, zoom=8, fmt="png"):
    """Convert all PDFs in a folder using PyMuPDF."""
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs found.")
        return
    for pdf in pdfs:
        pdf_to_images_pymupdf(os.path.join(folder_path, pdf), folder_path, zoom, fmt)
    print("\n[DONE] All conversions completed.")


if __name__ == "__main__":
    folder = r"D:\Windows Folders\Desktop\Hubble\static\images_static"
    convert_all_pdfs(folder, zoom=8, fmt="png")  # zoom=8 ≈ 600 DPI
