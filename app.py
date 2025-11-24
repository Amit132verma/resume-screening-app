import streamlit as st
import os
import re
import pdfplumber
import docx
import nltk
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# NLTK SETUP
# ---------------------------
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------------------
# SKILLS DB
# ---------------------------
skills_db = {
    "python", "java", "sql", "machine learning", "nlp", "deep learning",
    "excel", "c++", "cloud", "aws"
}

# ---------------------------
# SBERT MODEL
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')


# ---------------------------------------------------------
# TEXT EXTRACTION FUNCTIONS
# ---------------------------------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


def extract_name(text):
    match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", text)
    return match.group(0) if match else "Unknown"


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"


def extract_skills(text):
    words = set(text.split())
    return list(skills_db.intersection(words))


# ---------------------------------------------------------
# PROCESS A SINGLE RESUME
# ---------------------------------------------------------
def process_resume(uploaded_file):
    # Detect file type
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    else:
        return None

    cleaned = preprocess_text(text)

    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Skills": extract_skills(cleaned),
        "RawText": text
    }


# ---------------------------------------------------------
# RANK CANDIDATES USING SBERT
# ---------------------------------------------------------
def rank_candidates(job_description, resumes):
    cleaned_job_desc = preprocess_text(job_description)
    job_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)

    ranked = []

    for res in resumes:
        resume_embedding = model.encode(res["RawText"], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0].item()

        ranked.append({
            "Name": res["Name"],
            "Email": res["Email"],
            "Skills": res["Skills"],
            "Similarity": similarity
        })

    ranked = sorted(ranked, key=lambda x: x["Similarity"], reverse=True)
    return ranked[:5]  # Top 5


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Resume Screening App", layout="wide")

st.title("📄 Resume Screening System (SBERT)")
st.write("Upload multiple resumes + enter job description → get top 5 candidates.")

uploaded_files = st.file_uploader(
    "Upload multiple resumes (PDF/DOCX)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

job_description = st.text_area("Enter Job Description")

if st.button("Run Screening"):
    if not uploaded_files:
        st.error("Upload at least one resume.")
    elif not job_description.strip():
        st.error("Enter a job description.")
    else:
        resumes = []

        with st.spinner("Processing resumes..."):
            for f in uploaded_files:
                result = process_resume(f)
                if result:
                    resumes.append(result)

        st.success(f"Processed {len(resumes)} resumes ✔")

        with st.spinner("Ranking candidates..."):
            top5 = rank_candidates(job_description, resumes)

        st.subheader("🏆 Top 5 Candidates")

        for i, cand in enumerate(top5, 1):
            st.markdown(f"""
                <div style='padding:15px; border-radius:10px; border:1px solid #ccc; margin-bottom:15px;'>
                    <h3>{i}. {cand['Name']}</h3>
                    <b>Email:</b> {cand['Email']}<br>
                    <b>Skills:</b> {', '.join(cand['Skills'])}<br>
                    <b>Similarity Score:</b> {cand['Similarity']:.2f}
                </div>
            """, unsafe_allow_html=True)
