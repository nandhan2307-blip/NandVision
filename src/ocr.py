import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text(image_path):

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text


if __name__ == "__main__":

    image_path = "assets/testocr.png"

    text = extract_text(image_path)

    print("\n===== OCR RESULT =====")
    print(text)