from transformers import pipeline
from src.ocr import extract_text
from src.document import pdf_to_images


print("Loading document AI model...")

model = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

print("Document AI model loaded successfully!")


def find_relevant_context(question, document_text):

    lines = document_text.splitlines()

    keywords = question.lower().split()

    relevant_lines = []

    for line in lines:

        line_lower = line.lower()

        if any(keyword in line_lower for keyword in keywords):

            relevant_lines.append(line)

    return "\n".join(relevant_lines)


def ask_document(question, document_text):

    context = find_relevant_context(
        question,
        document_text
    )

    prompt = f"""
You are a document understanding assistant.

Answer the question using only the document text below.

DOCUMENT:
{context}

QUESTION:
{question}

ANSWER:
"""

    result = model(
        prompt,
        max_new_tokens=30,
        do_sample=False,
        return_full_text=False
    )

    answer = result[0]["generated_text"].strip()

    return answer


if __name__ == "__main__":

    pdf_path = "NandhanATS-resume.pdf"

    images = pdf_to_images(pdf_path)

    document_text = ""

    for image in images:

        page_text = extract_text(image)

        document_text += page_text + "\n"

    print("\n===== DOCUMENT TEXT EXTRACTED =====")
    print(document_text)

    # Ask a question
    question = "What is the name of the medicine stock visibility project?"

    answer = ask_document(
        question,
        document_text
    )

    print("\n===== DOCUMENT AI ANSWER =====")
    print(answer)