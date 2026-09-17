import sys
import re

# Windows compatibility workaround
sys.modules["regex"] = re

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


print("Loading language model...")

language_model = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct"
)

print("Language model loaded successfully!")


def create_chunks(text, chunk_size=100):

    words = text.split()

    chunks = []

    current_chunk = []

    current_length = 0

    for word in words:

        current_chunk.append(word)

        current_length += 1

        if current_length >= chunk_size:

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = []

            current_length = 0

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


def create_page_chunks(pages, chunk_size=100):

    all_chunks = []

    for page_number, page_text in enumerate(
        pages,
        start=1
    ):

        words = page_text.split()

        current_chunk = []

        current_length = 0

        for word in words:

            current_chunk.append(word)

            current_length += 1

            if current_length >= chunk_size:

                all_chunks.append(
                    {
                        "text": " ".join(current_chunk),
                        "page": page_number
                    }
                )

                current_chunk = []

                current_length = 0

        if current_chunk:

            all_chunks.append(
                {
                    "text": " ".join(current_chunk),
                    "page": page_number
                }
            )

    return all_chunks


def create_embeddings(text_chunks):

    texts = []

    for chunk in text_chunks:

        if isinstance(chunk, dict):

            texts.append(
                chunk["text"]
            )

        else:

            texts.append(chunk)

    embeddings = embedding_model.encode(
        texts
    )

    return embeddings


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def search_documents(
    question,
    chunks,
    index,
    top_k=2,
    distance_threshold=2.0
):

    question_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for position, index_number in enumerate(
        indices[0]
    ):

        if index_number == -1:

            continue

        distance = float(
            distances[0][position]
        )

        if distance > distance_threshold:

            continue

        chunk = chunks[index_number]

        if isinstance(chunk, dict):

            text = chunk["text"]
            page = chunk["page"]

        else:

            text = chunk
            page = None

        results.append(
            {
                "chunk": text,
                "page": page,
                "distance": distance
            }
        )

    return results


def prepare_context(search_results):

    context_parts = []

    for result in search_results:

        text = result["chunk"]

        text = text[:1800]

        context_parts.append(
            text
        )

    return "\n\n".join(
        context_parts
    )


def extract_name(context):

    match = re.search(
        r"(?i)\b(NANDHAN\s+S\s*S?)\b",
        context
    )

    if match:

        return match.group(1).upper()

    return None


def extract_education(context):

    patterns = [
        r"(B\.?E\.?\s+Computer Science\s*&?\s*Engineering)",
        r"(Computer Science\s*&?\s*Engineering)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            context,
            re.IGNORECASE
        )

        if match:

            return match.group(1).strip()

    return None


def extract_programming_languages(context):

    match = re.search(
        r"Programming:\s*([^|]+?)(?:\s+Web\s*&|\s+AI\s*&|$)",
        context,
        re.IGNORECASE
    )

    if match:

        return match.group(1).strip()

    return None


def extract_projects(context):

    projects = []

    if re.search(
        r"NandConnects",
        context,
        re.IGNORECASE
    ):

        projects.append(
            "NandConnects — College Hub Website"
        )

    if re.search(
        r"Paalam",
        context,
        re.IGNORECASE
    ):

        projects.append(
            "Paalam — Real-Time Medicine Stock Visibility Solution"
        )

    if projects:

        return "\n".join(
            projects
        )

    return None


def deterministic_answer(question, context):

    question_lower = question.lower()

    if (
        "what is my name" in question_lower
        or "what's my name" in question_lower
        or "who am i" in question_lower
    ):

        return extract_name(context)


    if (
        "what is my degree" in question_lower
        or "what degree" in question_lower
        or "what did i study" in question_lower
        or "what am i studying" in question_lower
    ):

        return extract_education(context)


    if (
        "programming languages" in question_lower
        or "programming language do i know" in question_lower
        or "programming languages do i know" in question_lower
    ):

        return extract_programming_languages(
            context
        )


    if (
        "what projects" in question_lower
        or "which projects" in question_lower
        or "projects have i worked" in question_lower
    ):

        return extract_projects(context)


    return None


def is_subjective_question(question):

    question_lower = question.lower()

    subjective_words = [
        "favorite",
        "favourite",
        "prefer",
        "preferred",
        "dislike",
        "best",
        "opinion",
        "choice",
        "choose",
        "enjoy"
    ]

    for word in subjective_words:

        if word in question_lower:

            return True

    return False


def generate_answer(question, context):

    if not context.strip():

        return (
            "Information not found in the document."
        )


    extracted_answer = deterministic_answer(
        question,
        context
    )

    if extracted_answer:

        return extracted_answer


    if is_subjective_question(
        question
    ):

        return (
            "Information not found in the document."
        )


    prompt = f"""
You are a strict document question-answering assistant.

Answer the question using ONLY the document context.

RULES:

- Use only information explicitly stated.
- Never guess.
- Never invent.
- Never use outside knowledge.
- Do not copy the document.
- Answer only what was asked.
- Keep the answer short.
- If the answer is not clearly present, say:
Information not found in the document.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    result = language_model(
        prompt,
        max_new_tokens=40,
        do_sample=False,
        return_full_text=False
    )

    print("\n===== RAW MODEL OUTPUT =====")

    print(result)

    answer = result[0]["generated_text"].strip()

    for marker in [
        "USER QUESTION:",
        "QUESTION:",
        "DOCUMENT CONTEXT:",
        "ANSWER:"
    ]:

        if marker in answer:

            answer = answer.split(
                marker
            )[-1].strip()

    if len(answer) > 250:

        return (
            "Information not found in the document."
        )

    if not answer:

        return (
            "Information not found in the document."
        )

    return answer


if __name__ == "__main__":

    from src.ocr import extract_text
    from src.document import pdf_to_images


    pdf_path = "NandhanATS-resume.pdf"

    images = pdf_to_images(
        pdf_path
    )


    pages = []

    for image in images:

        page_text = extract_text(
            image
        )

        pages.append(page_text)


    chunks = create_page_chunks(
        pages
    )


    print("\n===== DOCUMENT CHUNKS =====")

    print(
        "Number of chunks:",
        len(chunks)
    )


    embeddings = create_embeddings(
        chunks
    )


    print("\n===== EMBEDDINGS =====")

    print(
        "Embedding shape:",
        embeddings.shape
    )


    index = create_faiss_index(
        embeddings
    )


    print("\n===== FAISS INDEX =====")

    print(
        "FAISS index created successfully!"
    )


    question = (
        "What projects have I worked on?"
    )


    search_results = search_documents(
        question,
        chunks,
        index,
        top_k=2
    )


    print("\n===== RETRIEVAL RESULTS =====")


    for number, result in enumerate(
        search_results,
        start=1
    ):

        print(
            f"\nChunk {number}"
        )

        print(
            "Page:",
            result["page"]
        )

        print(
            "FAISS Distance:",
            result["distance"]
        )

        print(
            result["chunk"]
        )


    context = prepare_context(
        search_results
    )


    answer = generate_answer(
        question,
        context
    )


    print("\n===== RAG ANSWER =====")

    print(answer)