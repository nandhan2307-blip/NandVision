
import os
import tempfile

import streamlit as st

from src.vlm import describe_image
from src.detector import detect_objects
from src.ocr import extract_text
from src.document import pdf_to_images

from src.rag import (
    create_page_chunks,
    create_embeddings,
    create_faiss_index,
    search_documents,
    generate_answer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NANDVISION | AI Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #080b12;
        color: #f5f7fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d111b;
        border-right: 1px solid #202737;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    .block-container {
        max-width: 1500px;
        padding: 2.5rem 3rem 4rem 3rem;
    }

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .brand span {
        color: #f4c95d;
    }

    .muted {
        color: #8993a7;
        font-size: 13px;
    }

    .sidebar-divider {
        height: 1px;
        background: #252c3b;
        margin: 25px 0;
    }

    .sidebar-status {
        border: 1px solid #283143;
        background: #131925;
        border-radius: 14px;
        padding: 16px;
        margin-top: 30px;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #65d69b;
        margin-right: 7px;
    }

    .hero {
        background: linear-gradient(120deg, #151c2b 0%, #0d121e 100%);
        border: 1px solid #293246;
        border-radius: 24px;
        padding: 38px 42px;
        margin-bottom: 28px;
    }

    .hero-label {
        color: #f4c95d;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 52px);
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1.08;
        margin: 0;
        color: #ffffff;
    }

    .hero-description {
        max-width: 650px;
        margin-top: 18px;
        color: #a4aec0;
        font-size: 15px;
        line-height: 1.8;
    }

    .section-heading {
        font-size: 24px;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #8d98ab;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .feature-card {
        background: #111722;
        border: 1px solid #252e40;
        border-radius: 18px;
        padding: 22px;
        min-height: 145px;
        margin-bottom: 18px;
    }

    .feature-icon {
        font-size: 23px;
        margin-bottom: 12px;
    }

    .feature-title {
        font-size: 15px;
        font-weight: 700;
        color: #f5f7fb;
        margin-bottom: 8px;
    }

    .feature-description {
        color: #8e99ad;
        font-size: 12px;
        line-height: 1.6;
    }

    .workspace-banner {
        background: #121927;
        border: 1px solid #2b3549;
        border-radius: 18px;
        padding: 22px 25px;
        margin: 25px 0;
    }

    .workspace-title {
        font-size: 20px;
        font-weight: 750;
        color: #ffffff;
    }

    .workspace-subtitle {
        font-size: 13px;
        color: #929db0;
        margin-top: 5px;
    }

    .metric-card {
        background: #111722;
        border: 1px solid #283246;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .metric-number {
        color: #f4c95d;
        font-size: 27px;
        font-weight: 800;
    }

    .metric-label {
        color: #8f9aad;
        font-size: 12px;
        margin-top: 4px;
    }

    .result-card {
        background: #101722;
        border: 1px solid #2b3548;
        border-radius: 18px;
        padding: 25px;
        margin: 18px 0;
    }

    .result-heading {
        color: #f4c95d;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .info-pill {
        display: inline-block;
        background: #202a3b;
        color: #bdc6d5;
        border-radius: 20px;
        padding: 5px 11px;
        font-size: 11px;
        margin-right: 5px;
    }

    .stButton > button {
        background: #f4c95d;
        color: #10131a;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #ffdb7b;
        color: #10131a;
        border: none;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #3a465b;
        background: #171f2d;
        color: #ffffff;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: #111722;
        border: 1px solid #303b50;
        color: #ffffff;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: #111722;
        border: 1px dashed #3c485d;
        border-radius: 15px;
        padding: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0e141f;
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #8d98aa;
        border-radius: 8px;
        padding: 10px 18px;
    }

    .stTabs [aria-selected="true"] {
        background: #263044;
        color: #f4c95d;
    }

    .footer {
        text-align: center;
        color: #5f6b7f;
        font-size: 12px;
        margin-top: 55px;
        padding-top: 25px;
        border-top: 1px solid #202737;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">NAND<span>VISION</span></div>
        <div class="muted">Multimodal AI Workspace</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Navigation")

    workspace = st.radio(
        "Choose workspace",
        [
            "Image Intelligence",
            "Document Intelligence",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-status">
            <div style="font-weight:700; margin-bottom:12px;">
                System Status
            </div>
            <div style="font-size:12px; margin-bottom:9px;">
                <span class="status-dot"></span>AI Engine Ready
            </div>
            <div style="font-size:12px; margin-bottom:9px;">
                <span class="status-dot"></span>Vision Models Available
            </div>
            <div style="font-size:12px;">
                <span class="status-dot"></span>Document Pipeline Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("NANDVISION v1.0")
    st.caption("Built for intelligent visual understanding.")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">AI · VISION · KNOWLEDGE</div>
        <div class="hero-title">
            Intelligence,<br>
            beyond the screen.
        </div>
        <div class="hero-description">
            Analyze images, understand documents, extract information,
            and interact with intelligent AI systems through one unified workspace.
        </div>
        <br>
        <span class="info-pill">Computer Vision</span>
        <span class="info-pill">Multimodal AI</span>
        <span class="info-pill">Retrieval Augmented Generation</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">Workspace Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">Select a capability and start exploring.</div>',
    unsafe_allow_html=True,
)

overview_columns = st.columns(3)

overview_cards = [
    (
        "👁️",
        "Visual Understanding",
        "Ask questions about images using a vision-language model.",
    ),
    (
        "🎯",
        "Object Detection",
        "Detect and locate objects using YOLO-based computer vision.",
    ),
    (
        "📚",
        "Knowledge Retrieval",
        "Search documents and generate context-aware answers.",
    ),
]

for column, card in zip(overview_columns, overview_cards):
    with column:
        icon, title, description = card

        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# IMAGE INTELLIGENCE
# ============================================================

if workspace == "Image Intelligence":

    st.markdown(
        """
        <div class="workspace-banner">
            <div class="workspace-title">🖼️ Image Intelligence</div>
            <div class="workspace-subtitle">
                Understand visual content through AI-powered analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_uploader",
    )

    if image_file is None:
        st.info("Upload an image to activate the intelligence tools.")
    else:
        image_columns = st.columns([1.15, 1])

        with image_columns[0]:
            st.markdown("#### Input Preview")
            st.image(
                image_file,
                use_container_width=True,
                caption=image_file.name,
            )

        with image_columns[1]:
            st.markdown("#### Available Capabilities")

            st.markdown(
                """
                <div class="result-card">
                    <div class="result-heading">ACTIVE MODULES</div>
                    <span class="info-pill">Image Q&A</span>
                    <span class="info-pill">YOLO Detection</span>
                    <span class="info-pill">OCR</span>
                    <br><br>
                    <div class="muted">
                        Choose a module below to process your uploaded image.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        image_tab_qa, image_tab_detection, image_tab_ocr = st.tabs(
            [
                "💬 Image Q&A",
                "🎯 Object Detection",
                "🔤 OCR Extraction",
            ]
        )

        with image_tab_qa:
            st.markdown("### Ask your image")

            question = st.text_input(
                "What would you like to know?",
                placeholder="Describe the image, identify objects, or ask a question...",
                key="image_question",
            )

            if st.button("Analyze Image", key="analyze_image"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Analyzing image with the vision model..."):
                        try:
                            answer = describe_image(
                                image_file,
                                question,
                            )

                            st.markdown(
                                """
                                <div class="result-card">
                                    <div class="result-heading">
                                        VISION MODEL RESPONSE
                                    </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.write(answer)

                            st.markdown("</div>", unsafe_allow_html=True)

                        except Exception as error:
                            st.error(f"Image analysis failed: {error}")

        with image_tab_detection:
            st.markdown("### Detect objects")

            if st.button("Run Object Detection", key="run_detection"):
                with st.spinner("Detecting objects..."):
                    try:
                        detected_image, detections = detect_objects(image_file)

                        st.image(
                            detected_image,
                            use_container_width=True,
                            caption="Detected objects",
                        )

                        st.markdown(
                            '<div class="result-heading">DETECTION RESULTS</div>',
                            unsafe_allow_html=True,
                        )

                        if detections:
                            for detection in detections:
                                st.write(detection)
                        else:
                            st.info("No objects were detected.")

                    except Exception as error:
                        st.error(f"Object detection failed: {error}")

        with image_tab_ocr:
            st.markdown("### Extract text from image")

            if st.button("Extract Text", key="extract_ocr"):
                with st.spinner("Extracting text using OCR..."):
                    try:
                        extracted_text = extract_text(image_file)

                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="result-heading">
                                    EXTRACTED TEXT
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if extracted_text and extracted_text.strip():
                            st.text_area(
                                "OCR Output",
                                extracted_text,
                                height=260,
                                key="ocr_output",
                            )
                        else:
                            st.info("No readable text was found.")

                        st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as error:
                        st.error(f"OCR extraction failed: {error}")


# ============================================================
# DOCUMENT INTELLIGENCE
# ============================================================

elif workspace == "Document Intelligence":

    st.markdown(
        """
        <div class="workspace-banner">
            <div class="workspace-title">📄 Document Intelligence</div>
            <div class="workspace-subtitle">
                Convert documents into searchable knowledge using RAG.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="pdf_uploader",
    )

    if pdf_file is None:
        st.info("Upload a PDF document to begin processing.")

    else:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-heading">DOCUMENT SELECTED</div>
                <strong>{pdf_file.name}</strong>
                <br><br>
                <span class="info-pill">PDF</span>
                <span class="info-pill">OCR</span>
                <span class="info-pill">Semantic Search</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Process Document", key="process_document"):

            with st.spinner("Processing document..."):

                temporary_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as temporary_file:
                        temporary_file.write(pdf_file.getbuffer())
                        temporary_path = temporary_file.name

                    pages = pdf_to_images(temporary_path)

                    page_chunks = create_page_chunks(pages)

                    embeddings = create_embeddings(page_chunks)

                    faiss_index = create_faiss_index(embeddings)

                    st.session_state.document_chunks = page_chunks
                    st.session_state.document_embeddings = embeddings
                    st.session_state.document_index = faiss_index
                    st.session_state.document_pages = pages
                    st.session_state.document_name = pdf_file.name

                    st.success("Document processed successfully.")

                except Exception as error:
                    st.error(f"Document processing failed: {error}")

                finally:
                    if temporary_path and os.path.exists(temporary_path):
                        os.remove(temporary_path)

        if "document_index" in st.session_state:

            st.markdown("---")

            metric_columns = st.columns(3)

            with metric_columns[0]:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-number">✓</div>
                        <div class="metric-label">Document Processed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_columns[1]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {len(st.session_state.get("document_pages", []))}
                        </div>
                        <div class="metric-label">Pages</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric_columns[2]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">
                            {len(st.session_state.get("document_chunks", []))}
                        </div>
                        <div class="metric-label">Knowledge Chunks</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="workspace-banner">
                    <div class="workspace-title">
                        💬 Ask your document
                    </div>
                    <div class="workspace-subtitle">
                        Ask questions and retrieve relevant information from your PDF.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            document_question = st.text_input(
                "Enter your question",
                placeholder="What problem does this document discuss?",
                key="document_question",
            )

            result_count = st.slider(
                "Number of relevant passages",
                min_value=1,
                max_value=8,
                value=3,
                key="result_count",
            )

            if st.button("Ask Document", key="ask_document"):

                if not document_question.strip():
                    st.warning("Please enter a question.")

                else:
                    with st.spinner("Searching document knowledge..."):

                        try:
                            retrieved_results = search_documents(
                                document_question,
                                st.session_state.document_index,
                                st.session_state.document_embeddings,
                                st.session_state.document_chunks,
                                top_k=result_count,
                            )

                            answer = generate_answer(
                                document_question,
                                retrieved_results,
                            )

                            st.markdown(
                                """
                                <div class="result-card">
                                    <div class="result-heading">
                                        AI-GENERATED ANSWER
                                    </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.write(answer)

                            st.markdown("</div>", unsafe_allow_html=True)

                            st.markdown("### Retrieved Context")

                            if retrieved_results:

                                for index, result in enumerate(
                                    retrieved_results,
                                    start=1,
                                ):

                                    chunk_text = result.get(
                                        "text",
                                        result.get("chunk", ""),
                                    )

                                    page_number = result.get(
                                        "page",
                                        "Unknown",
                                    )

                                    distance = result.get(
                                        "distance",
                                        "N/A",
                                    )

                                    with st.expander(
                                        f"Passage {index} · Page {page_number}"
                                    ):

                                        st.write(chunk_text)

                                        st.caption(
                                            f"Similarity distance: {distance}"
                                        )

                            else:
                                st.info("No relevant passages were found.")

                        except Exception as error:
                            st.error(
                                f"Document question answering failed: {error}"
                            )

            st.markdown("---")

            with st.expander("View document pages"):

                document_pages = st.session_state.get(
                    "document_pages",
                    [],
                )

                if document_pages:

                    for page_number, page in enumerate(
                        document_pages,
                        start=1,
                    ):

                        st.markdown(f"#### Page {page_number}")
                        st.image(
                            page,
                            use_container_width=True,
                        )

                else:
                    st.info("No document pages available.")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NANDVISION · Multimodal AI Intelligence Workspace
        <br>
        Built with Streamlit, Computer Vision, Transformers, and RAG.
    </div>
    """,
    unsafe_allow_html=True,
)