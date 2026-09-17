
import os
import tempfile
import textwrap

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


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NANDVISION",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #263244;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        .hero-title {
            font-size: 42px;
            font-weight: 750;
            letter-spacing: -1.5px;
            color: #f9fafb;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            font-size: 17px;
            color: #9ca3af;
            margin-bottom: 2rem;
        }

        .section-title {
            font-size: 27px;
            font-weight: 700;
            color: #f3f4f6;
            margin-top: 1rem;
            margin-bottom: 0.3rem;
        }

        .section-subtitle {
            font-size: 15px;
            color: #9ca3af;
            margin-bottom: 1.5rem;
        }

        .feature-card {
            background: linear-gradient(
                145deg,
                #1b2433,
                #141b27
            );
            border: 1px solid #2c394d;
            border-radius: 14px;
            padding: 22px;
            min-height: 145px;
            margin-bottom: 18px;
        }

        .feature-card h3 {
            font-size: 19px;
            color: #f9fafb;
            margin-bottom: 8px;
        }

        .feature-card p {
            font-size: 14px;
            color: #aeb8c7;
            line-height: 1.6;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    # Native Streamlit components — prevents HTML display issue

    st.markdown("## ◈ NANDVISION")

    st.caption("Multimodal AI Intelligence Workspace")

    st.divider()

    st.markdown("#### WORKSPACE")

    selected_section = st.radio(
        "Select workspace",
        [
            "Image Intelligence",
            "Document Intelligence",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("#### TECHNOLOGY")

    st.caption("Python")
    st.caption("PyTorch")
    st.caption("YOLO")
    st.caption("OCR")
    st.caption("FAISS / RAG")

    st.divider()

    st.caption("NANDVISION")
    st.caption("Portfolio Project")


# ==================================================
# HERO HEADER
# ==================================================

st.markdown(
    '<div class="hero-title">NANDVISION</div>',
    unsafe_allow_html=True,
)

st.markdown(
    textwrap.dedent(
        """
        <div class="hero-subtitle">
            Analyze visual content, extract knowledge, and interact
            with intelligent document systems.
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ==================================================
# IMAGE INTELLIGENCE
# ==================================================

if selected_section == "Image Intelligence":

    st.markdown(
        '<div class="section-title">Image Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="section-subtitle">
                Understand images, detect objects, and extract
                visual text using AI-powered tools.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Visual Understanding</h3>
                    <p>
                        Ask questions about uploaded images
                        using a vision-language model.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with feature_col2:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Object Detection</h3>
                    <p>
                        Identify and visualize objects
                        using YOLO computer vision.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with feature_col3:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Text Extraction</h3>
                    <p>
                        Extract readable text from images
                        using OCR technology.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True,
        )

        image_suffix = os.path.splitext(
            uploaded_image.name
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=image_suffix,
        ) as temp_file:

            temp_file.write(
                uploaded_image.getbuffer()
            )

            image_path = temp_file.name

        st.divider()

        image_tab1, image_tab2, image_tab3 = st.tabs(
            [
                "Image Q&A",
                "Object Detection",
                "OCR",
            ]
        )

        # ==========================================
        # IMAGE Q&A
        # ==========================================

        with image_tab1:

            st.subheader("Ask about your image")

            question = st.text_input(
                "Enter your question",
                placeholder="What can you see in this image?",
                key="image_question",
            )

            if question:

                with st.spinner("Analyzing visual content..."):

                    try:

                        result = describe_image(
                            image_path,
                            question,
                        )

                        st.markdown("#### AI Response")
                        st.info(result)

                    except Exception as error:

                        st.error(
                            f"Image analysis failed: {error}"
                        )

        # ==========================================
        # OBJECT DETECTION
        # ==========================================

        with image_tab2:

            st.subheader("Detect objects")

            st.write(
                "Identify objects present in the uploaded image."
            )

            if st.button(
                "Run Object Detection",
                key="detect_button",
                use_container_width=True,
            ):

                with st.spinner("Running YOLO detection..."):

                    try:

                        annotated_image = detect_objects(
                            image_path
                        )

                        st.image(
                            annotated_image,
                            channels="BGR",
                            caption="Detection Results",
                            use_container_width=True,
                        )

                    except Exception as error:

                        st.error(
                            f"Object detection failed: {error}"
                        )

        # ==========================================
        # OCR
        # ==========================================

        with image_tab3:

            st.subheader("Extract image text")

            st.write(
                "Convert text visible in an image into editable text."
            )

            if st.button(
                "Extract Text",
                key="ocr_button",
                use_container_width=True,
            ):

                with st.spinner("Extracting text..."):

                    try:

                        text = extract_text(image_path)

                        if text.strip():

                            st.text_area(
                                "Extracted Text",
                                text,
                                height=250,
                            )

                        else:

                            st.info(
                                "No readable text detected."
                            )

                    except Exception as error:

                        st.error(
                            f"OCR failed: {error}"
                        )

        try:

            os.remove(image_path)

        except OSError:

            pass


# ==================================================
# DOCUMENT INTELLIGENCE
# ==================================================

if selected_section == "Document Intelligence":

    st.markdown(
        '<div class="section-title">Document Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="section-subtitle">
                Process PDF documents, retrieve relevant information,
                and ask questions using retrieval-augmented generation.
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Document Processing</h3>
                    <p>
                        Convert PDF pages into readable text
                        using OCR.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with feature_col2:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Semantic Retrieval</h3>
                    <p>
                        Search document content using
                        embeddings and FAISS.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with feature_col3:

        st.markdown(
            textwrap.dedent(
                """
                <div class="feature-card">
                    <h3>Grounded Answers</h3>
                    <p>
                        Generate answers using relevant
                        document context.
                    </p>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            temp_file.write(
                uploaded_pdf.getbuffer()
            )

            pdf_path = temp_file.name

        if st.button(
            "Process Document",
            key="process_document_button",
            use_container_width=True,
        ):

            with st.spinner(
                "Processing document and building search index..."
            ):

                try:

                    images = pdf_to_images(pdf_path)

                    pages = []

                    for image in images:

                        page_text = extract_text(image)
                        pages.append(page_text)

                    chunks = create_page_chunks(
                        pages,
                        chunk_size=300,
                    )

                    if not chunks:

                        st.error(
                            "No readable text found in the document."
                        )

                    else:

                        embeddings = create_embeddings(chunks)

                        index = create_faiss_index(embeddings)

                        st.session_state["document_chunks"] = chunks
                        st.session_state["document_index"] = index
                        st.session_state["document_pages"] = pages
                        st.session_state["document_name"] = uploaded_pdf.name

                        st.success(
                            "Document processed successfully."
                        )

                        metric_col1, metric_col2 = st.columns(2)

                        with metric_col1:

                            st.metric("Pages", len(pages))

                        with metric_col2:

                            st.metric("Search Chunks", len(chunks))

                except Exception as error:

                    st.error(
                        f"Document processing failed: {error}"
                    )

                finally:

                    try:

                        os.remove(pdf_path)

                    except OSError:

                        pass

    # ==============================================
    # DOCUMENT QUESTIONS
    # ==============================================

    if (
        "document_chunks" in st.session_state
        and "document_index" in st.session_state
    ):

        st.divider()

        st.subheader("Ask your document")

        st.caption(
            f"Active document: "
            f"{st.session_state.get('document_name', 'Uploaded PDF')}"
        )

        document_question = st.text_input(
            "Enter your question",
            placeholder="What problem does Paalam solve?",
            key="document_question",
        )

        if document_question:

            with st.spinner("Searching document knowledge..."):

                try:

                    search_results = search_documents(
                        document_question,
                        st.session_state["document_chunks"],
                        st.session_state["document_index"],
                        top_k=2,
                    )

                    if not search_results:

                        st.info(
                            "Information not found in the document."
                        )

                    else:

                        context = "\n".join(
                            result["chunk"]
                            for result in search_results
                        )

                        answer = generate_answer(
                            document_question,
                            context,
                        )

                        st.markdown("#### AI Response")

                        st.info(answer)

                        with st.expander("View retrieval details"):

                            for number, result in enumerate(
                                search_results,
                                start=1,
                            ):

                                st.markdown(
                                    f"**Retrieved Chunk {number}**"
                                )

                                if result["page"] is not None:

                                    st.write(
                                        f"Source Page: {result['page']}"
                                    )

                                st.write(
                                    f"FAISS Distance: "
                                    f"{result['distance']:.4f}"
                                )

                                with st.expander(
                                    f"View source chunk {number}"
                                ):

                                    st.write(result["chunk"])

                                st.divider()

                except Exception as error:

                    st.error(
                        f"Question answering failed: {error}"
                    )