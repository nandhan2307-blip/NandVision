import fitz


def pdf_to_images(pdf_path):

    pdf = fitz.open(pdf_path)

    images = []

    for page_number, page in enumerate(pdf):

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        image_path = f"assets/document_page_{page_number + 1}.png"

        pix.save(image_path)

        images.append(image_path)

    pdf.close()

    return images


if __name__ == "__main__":

    pdf_path = "NandhanATS-resume.pdf"

    images = pdf_to_images(pdf_path)

    print("Number of pages:", len(images))

    for image in images:
        print("Page image saved to:", image)