
import os
import shutil

import pytesseract
from PIL import Image


# Configure Tesseract for local Windows or Streamlit Cloud
def configure_tesseract():
    windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if os.path.exists(windows_path):
        # Local Windows configuration
        pytesseract.pytesseract.tesseract_cmd = windows_path

    elif shutil.which("tesseract"):
        # Streamlit Cloud / Linux configuration
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

    else:
        raise RuntimeError(
            "Tesseract OCR is not installed or cannot be found."
        )


def extract_text(image_path):
    configure_tesseract()

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text


if __name__ == "__main__":
    image_path = "assets/testocr.png"

    text = extract_text(image_path)

    print("\n===== OCR RESULT =====")
    print(text)