import streamlit as st
import pandas as pd
import os
from preprocessing import preprocess_text
from search_methods import document_term_incidence, inverted_index_search, tfidf_search, cosine_similarity_search

DATA_FILE = "saved_sentences.csv"  # ملف حفظ الجمل

def save_documents_to_file(documents):
    df = pd.DataFrame(list(documents.items()), columns=["doc_id", "text"])
    df.to_csv(DATA_FILE, index=False)

def load_documents_from_file():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # تأكد من تحويل المفاتيح لـ int لأن في CSV ممكن يكونوا محفوظين كـ string
        return {int(row.doc_id): row.text for row in df.itertuples()}
    return {}

st.set_page_config(
    page_title="Simple Search Engine",
    layout="centered",
    initial_sidebar_state="expanded"
)

# خلفية بيضاء
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "welcome"

def go_to_main():
    st.session_state.page = "main"

def go_to_welcome():
    st.session_state.page = "welcome"

# تحميل الجمل من الملف عند بدء التطبيق
if "documents" not in st.session_state:
    st.session_state.documents = load_documents_from_file()
    if st.session_state.documents:
        st.session_state.doc_id = max(st.session_state.documents.keys()) + 1
    else:
        st.session_state.doc_id = 1

if st.session_state.page == "welcome":
    st.title("Welcome to the Simple Search Engine")

    st.markdown(
        """
        <div style='font-size: 18px; margin-top: 20px; margin-bottom: 30px; line-height: 1.6;'>
            This tool helps you search across your custom sentences using different 
            <strong>Information Retrieval</strong> methods.
            <br><br>
            Choose your method and get results instantly.
        </div>
        """,
        unsafe_allow_html=True
    )

    center_col = st.columns(3)[1]
    with center_col:
        if st.button("Go to Search Engine"):
            go_to_main()

elif st.session_state.page == "main":
    st.title("Simple Search Engine")

    # إدخال جملة جديدة
    new_sentence = st.text_input("Enter a new sentence:")
    if st.button("Add Sentence"):
        if new_sentence.strip():
            st.session_state.documents[st.session_state.doc_id] = new_sentence.strip()
            st.session_state.doc_id += 1
            save_documents_to_file(st.session_state.documents)  # حفظ تلقائي بعد الإضافة
            st.success("Sentence added and saved successfully.")
        else:
            st.warning("Please enter a non-empty sentence.")

    # عرض الجمل مع زر حذف لكل جملة
    if st.session_state.documents:
        st.subheader("Entered Sentences")
        to_delete = None
        for doc_id, text in st.session_state.documents.items():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f"**Document {doc_id}**: {text}")
            with col2:
                if st.button("Delete", key=f"delete_{doc_id}"):
                    to_delete = doc_id
        if to_delete:
            del st.session_state.documents[to_delete]
            save_documents_to_file(st.session_state.documents)  # تحديث الملف بعد الحذف
            st.success(f"Deleted Document {to_delete}")
    else:
        st.info("No sentences added yet.")

    # المعالجة المسبقة
    if st.session_state.documents:
        preprocessed_docs = {
            doc_id: preprocess_text(text)
            for doc_id, text in st.session_state.documents.items()
        }
        st.subheader("Preprocessed Sentences")
        for doc_id, tokens in preprocessed_docs.items():
            st.write(f"**Document {doc_id}**: {tokens}")
    else:
        preprocessed_docs = {}

    # البحث
    query = st.text_input("Enter search query")
    methods = st.multiselect(
        "Select search methods",
        ["Document-Term Incidence", "Inverted Index", "TF-IDF", "Cosine Similarity"],
    )

    if st.button("Start Processing"):
        if not st.session_state.documents:
            st.warning("No sentences available for processing.")
        elif not query.strip():
            st.warning("Please enter the search query.")
        elif not methods:
            st.warning("Please select at least one search method.")
        else:
            st.subheader("Processing Results")

            if "Document-Term Incidence" in methods:
                st.markdown("### Document-Term Incidence")
                result = document_term_incidence(preprocessed_docs, query)
                if isinstance(result, list):
                    data = [{"Document ID": doc_id, "Document Text": st.session_state.documents[doc_id]} for doc_id in result]
                    st.table(pd.DataFrame(data))
                else:
                    st.write(result)

            if "Inverted Index" in methods:
                st.markdown("### Inverted Index")
                result = inverted_index_search(preprocessed_docs, query)
                if isinstance(result, list):
                    data = [{"Document ID": doc_id, "Document Text": st.session_state.documents[doc_id]} for doc_id in result]
                    st.table(pd.DataFrame(data))
                else:
                    st.write(result)

            if "TF-IDF" in methods:
                st.markdown("### TF-IDF")
                result = tfidf_search(preprocessed_docs, query)
                if isinstance(result, dict) and result:
                    data = [{"Document ID": doc_id, "Score": round(score, 4), "Document Text": st.session_state.documents[doc_id]} for doc_id, score in result.items()]
                    df = pd.DataFrame(data).sort_values(by="Score", ascending=False)
                    st.dataframe(df)
                else:
                    st.info("No TF-IDF scores available.")

            if "Cosine Similarity" in methods:
                st.markdown("### Cosine Similarity")
                result = cosine_similarity_search(preprocessed_docs, query)
                if isinstance(result, dict) and result:
                    data = [{"Document ID": doc_id, "Similarity": round(score, 4), "Document Text": st.session_state.documents[doc_id]} for doc_id, score in result.items()]
                    df = pd.DataFrame(data).sort_values(by="Similarity", ascending=False)
                    st.dataframe(df)
                else:
                    st.info("No cosine similarity scores available.")

    # فراغات لدفع زر الرجوع لأسفل يمين الصفحة
    st.write("\n" * 40)

    _, right_col = st.columns([9, 1])
    with right_col:
        if st.button("Back to Welcome"):
            go_to_welcome()

        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #4CAF50;
                color: white;
                padding: 3px 5px;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
                transition-duration: 0.4s;
                width: 100%;
            }
            div.stButton > button:first-child:hover {
                background-color: #45a049;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
