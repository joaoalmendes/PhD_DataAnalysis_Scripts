import os
import argparse
from pdf2image import convert_from_path

def convert_pdfs_to_png(input_folder, output_folder=None, dpi=600):
    input_folder = os.path.abspath(input_folder)
    if output_folder is None:
        output_folder = os.path.join(input_folder, "pngs")
    else:
        output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, pdf_file)
            pages = convert_from_path(pdf_path, dpi=dpi)

            base_name = os.path.splitext(pdf_file)[0]

            if len(pages) == 1:
                # Single-page PDF
                output_path = os.path.join(output_folder, f"{base_name}.png")
                pages[0].save(output_path, "PNG")
            else:
                # Multi-page PDF
                for i, page in enumerate(pages):
                    output_path = os.path.join(output_folder, f"{base_name}_page{i+1}.png")
                    page.save(output_path, "PNG")

    print(f"Done. PNGs saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all PDFs in a folder to PNG images.")
    parser.add_argument("input_folder", help="Path to the folder containing PDF files")
    parser.add_argument("-o", "--output_folder", default=None,
                         help="Path to save output PNGs (default: <input_folder>/pngs)")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for conversion (default: 600)")
    args = parser.parse_args()

    convert_pdfs_to_png(args.input_folder, args.output_folder, args.dpi)