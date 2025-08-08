import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import docx
import string
from pdfminer.high_level import extract_text as extract_pdf_text

st.set_page_config(page_title="Resume Screening App", layout="centered")
st.title("📄 Resume Screening App")
st.write("Upload a job description and one or more resumes to see how well they match.")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper functions ---

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_pdf_text(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    else:
        return uploaded_file.read().decode('utf-8')

def get_keywords(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    common_words = {'and', 'or', 'the', 'a', 'an', 'in', 'of', 'with', 'to', 'for', 'on', 'at', 'as', 'is', 'are', 'we', 'you', 'they', 'this'}
    keywords = [w for w in words if len(w) > 2 and w not in common_words]
    return set(keywords)

# --- Sidebar instructions ---
with st.sidebar:
    st.header("📌 Instructions")
    st.markdown("""
    1. Upload a **job description** in `.txt`, `.pdf`, or `.docx` format.
    2. Upload **one or more resumes** in the same formats.
    3. View match scores and justifications below.
    """)

# --- Uploads ---
st.subheader("1. Upload Job Description")
job_file = st.file_uploader("Upload job description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

st.subheader("2. Upload Resumes")
resume_files = st.file_uploader("Upload one or more resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

# --- Processing ---
if job_file and resume_files:
    job_description = extract_text_from_uploaded_file(job_file)
    job_embedding = model.encode([job_description])
    job_keywords = get_keywords(job_description)

    results = []

    for file in resume_files:
        resume_text = extract_text_from_uploaded_file(file)
        resume_embedding = model.encode([resume_text])
        score = cosine_similarity(resume_embedding, job_embedding)[0][0]
        resume_keywords = get_keywords(resume_text)
        matched_keywords = job_keywords.intersection(resume_keywords)

        results.append({
            "Resume Name": file.name,
            "Match Score (%)": round(score * 100, 2),
            "Excerpt": resume_text[:300] + "...",
            "Justification": ", ".join(sorted(matched_keywords)[:5]) + "..." if matched_keywords else "No strong keyword matches."
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Match Score (%)", ascending=False)

    # Output
    st.subheader("🔍 Matching Results")
    st.dataframe(results_df[["Resume Name", "Match Score (%)"]], use_container_width=True)

    with st.expander("📄 See Resume Details & Justifications"):
        for _, row in results_df.iterrows():
            st.markdown(f"**{row['Resume Name']}** — *{row['Match Score (%)']}%*")
            st.code(row["Excerpt"])
            st.markdown(f"**Justification:** {row['Justification']}")
            st.markdown("---")
else:
    st.info("Please upload both a job description and at least one resume.")
