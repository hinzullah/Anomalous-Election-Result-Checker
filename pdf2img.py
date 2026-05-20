import os
from pdf2image import convert_from_path
from PIL import Image

def convert_pdfs_to_images(pdf_folder, image_output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    # List all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"Converting {pdf_path} to images...")

        # Convert PDF to images
        try:
            # You can specify the poppler path if needed
            poppler_path = r"C:\Release-24.02.0-0[1]\poppler-24.02.0\Library\bin"  # Update this path for your system
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
            base_filename = os.path.splitext(pdf_file)[0]

            for i, image in enumerate(images):
                image_filename = f"{base_filename}_page_{i + 1}.jpg"
                image_path = os.path.join(image_output_folder, image_filename)
                image.save(image_path, 'JPEG')
                print(f"Saved {image_path}")
        
        except Exception as e:
            print(f"Error converting {pdf_path}: {e}")

# Example usage
pdf_folder = r"C:\Users\DONKAMS\Desktop\Final year project\pdfs"  # Folder with PDFs
image_output_folder = r"C:\Users\DONKAMS\Desktop\Final year project\images"  # Folder to save images

convert_pdfs_to_images(pdf_folder, image_output_folder)
