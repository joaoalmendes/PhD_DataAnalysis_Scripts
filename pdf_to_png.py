from pdf2image import convert_from_path
import os

pdf_folder = os.getcwd() #"RSM_cuts_pdf/L_cuts/Hysteric/NewPeaks"
output_folder = os.getcwd() #"RSM_cuts/L_cuts/Hysteric/NewPeaks"
os.makedirs(output_folder, exist_ok=True)

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pages = convert_from_path(pdf_path, dpi=600)
        
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
