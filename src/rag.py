
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


def create_chunks(text, chunk_size=300):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_page_chunks(pages, chunk_size=300):

    all_chunks = []

    for page_number, page_text in enumerate(pages, start=1):

        chunks = create_chunks(page_text, chunk_size)

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page_number
            })

    return all_chunks


def create_embeddings(chunks):

    texts = [
        chunk["text"] if isinstance(chunk, dict) else chunk
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings.astype("float32"))

    return index


def search_documents(
    question,
    chunks,
    index,
    top_k=3,
    distance_threshold=2.0
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        if distance > distance_threshold:
            continue

        result = chunks[index_position].copy()

        result["distance"] = float(distance)

        results.append(result)

    return results


def prepare_context(results):

    context_parts = []

    for result in results:

        page = result.get("page", "Unknown")

        text = result.get("text", "")

        context_parts.append(
            f"[Page {page}]\n{text}"
        )

    return "\n\n".join(context_parts)


def extract_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def find_project_answer(context, project_name):

    context_lower = context.lower()

    project_position = context_lower.find(
        project_name.lower()
    )

    if project_position == -1:
        return None

    start = max(0, project_position - 150)

    end = min(
        len(context),
        project_position + 500
    )

    answer = context[start:end].strip()

    return answer


def deterministic_answer(question, context):

    question_lower = question.lower()

    # Name-related questions
    if "what is my name" in question_lower or (
        "name" in question_lower
        and "nandhan" in context.lower()
    ):

        name_match = re.search(
            r"(?:name\s*[:\-]?\s*)([A-Za-z ]{2,40})",
            context,
            re.IGNORECASE
        )

        if name_match:

            name = name_match.group(1).strip()

            return f"The name mentioned in the document is {name}."

    # Education-related questions
    if any(
        keyword in question_lower
        for keyword in [
            "education",
            "educational qualification",
            "degree",
            "college"
        ]
    ):

        sentences = extract_sentences(context)

        education_sentences = [
            sentence
            for sentence in sentences
            if any(
                keyword in sentence.lower()
                for keyword in [
                    "bachelor",
                    "engineering",
                    "college",
                    "university",
                    "degree",
                    "computer science"
                ]
            )
        ]

        if education_sentences:

            return " ".join(education_sentences[:3])

    # Programming language questions
    if (
        "programming language" in question_lower
        or "coding language" in question_lower
    ):

        language_names = [
            "Python",
            "C",
            "C++",
            "Java",
            "JavaScript",
            "HTML",
            "CSS",
            "SQL"
        ]

        found_languages = []

        for language in language_names:

            if re.search(
                rf"\b{re.escape(language)}\b",
                context,
                re.IGNORECASE
            ):

                if language not in found_languages:

                    found_languages.append(language)

        if found_languages:

            return (
                "The programming languages mentioned "
                "in the document are: "
                + ", ".join(found_languages)
                + "."
            )

    # Project-related questions
    if "project" in question_lower:

        project_names = [
            "Paalam",
            "NandConnects"
        ]

        found_projects = []

        for project in project_names:

            if project.lower() in context.lower():

                found_projects.append(project)

        if found_projects:

            return (
                "The projects mentioned in the document "
                "include: "
                + ", ".join(found_projects)
                + "."
            )

    # Paalam-related questions
    if "paalam" in question_lower:

        answer = find_project_answer(
            context,
            "Paalam"
        )

        if answer:

            return answer

    # NandConnects-related questions
    if "nandconnects" in question_lower:

        answer = find_project_answer(
            context,
            "NandConnects"
        )

        if answer:

            return answer

    return None


def is_subjective_question(question):

    subjective_keywords = [
        "explain",
        "describe",
        "summarize",
        "why",
        "how",
        "purpose",
        "problem",
        "impact"
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in subjective_keywords
    )


def clean_model_answer(answer, question):

    answer = answer.strip()

    answer = re.sub(
        r"^(Answer|Response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = answer.split("<|im_end|>")[0].strip()

    answer = answer.split("<|endoftext|>")[0].strip()

    answer = re.sub(
        r"\s+",
        " ",
        answer
    ).strip()

    question_clean = re.sub(
        r"\s+",
        " ",
        question
    ).strip().lower()

    answer_clean = answer.lower()

    if answer_clean == question_clean:

        return ""

    if len(answer) > 500:

        sentences = extract_sentences(answer)

        answer = " ".join(sentences[:4])

    return answer.strip()


def generate_answer(question, context):

    if not context.strip():

        return (
            "I could not find relevant information "
            "in the document."
        )

    deterministic_result = deterministic_answer(
        question,
        context
    )

    if deterministic_result:

        return deterministic_result

    prompt = f"""<|im_start|>system
You are a document question-answering assistant.

Answer the user's question using only the provided document context.

Do not repeat the question.
Do not invent information.
Give a short, clear and direct answer.
If the context does not contain the answer, say that the information is not available.
<|im_end|>
<|im_start|>user
Document context:

{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

    try:

        output = language_model(
            prompt,
            max_new_tokens=60,
            do_sample=False,
            return_full_text=False
        )

        generated_text = output[0]["generated_text"]

        answer = clean_model_answer(
            generated_text,
            question
        )

        if not answer:

            return (
                "I could not generate a clear answer "
                "from the document."
            )

        return answer

    except Exception as error:

        print("Answer generation error:", error)

        return (
            "An error occurred while generating "
            "the answer."
        )


if __name__ == "__main__":

    print("\nRAG module loaded successfully!")

    sample_context = """
    Paalam is a real-time medicine stock visibility
    solution for rural PHC clusters. It helps ensure
    the right medicine reaches the right hands at
    the right time.
    """

    sample_question = "What problem does Paalam solve?"

    print("\n===== TEST QUESTION =====")
    print(sample_question)

    print("\n===== TEST ANSWER =====")

    answer = generate_answer(
        sample_question,
        sample_context
    )

    print(answer)