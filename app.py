import os
import re
import nltk
import docx
import pdfplumber
import pandas as pd
import streamlit as st
import subprocess

# Sentence-Transformers
from sentence_transformers import SentenceTransformer, util

# ✅ Ensure NLTK Dependencies
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))

# ✅ spaCy for Name Extraction
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    nlp = None
    st.warning("⚠️ spaCy not installed. Name extraction will fall back to regex.")

# ✅ Skills Database
skills_db = {
    "python", "java", "sql", "machine learning", "nlp",
    "deep learning", "excel", "c++", "cloud", "aws",
    "spring", "hibernate", "angular", "node.js", "html", "css", "javascript",
    "react", "mongodb", "postgresql", "rest api"
}

# ------------------- Extract Text ------------------- #
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        st.error(f"❌ Error reading PDF {pdf_path}: {e}")
    return text.strip()

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"❌ Error reading DOCX {docx_path}: {e}")
        return ""

# ------------------- Preprocessing ------------------- #
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception:
        return text

# ------------------- Information Extraction ------------------- #
def extract_name(text):
    """Try spaCy PERSON entity, else regex fallback."""
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text.strip()
        except Exception:
            pass

    # Regex fallback
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$)',
        r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})(?:\s*\n|\s*$)',
        r'([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})(?:\s*\n|\s*$)'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return "Unknown"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in skills_db if skill in text_lower]

# ------------------- Resume Processing ------------------- #
def process_resumes(resume_folder):
    resume_data = []

    if not os.path.exists(resume_folder):
        st.error(f"❌ Folder not found: {resume_folder}")
        return []

    for file_name in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file_name)

        if file_name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_name.lower().endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            continue

        if not text.strip():
            continue

        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)

        resume_data.append({
            "Name": name,
            "Email": email,
            "Skills": skills,
            "RawText": text
        })

    return resume_data

# ------------------- Streamlit App ------------------- #
def main():
    st.title("📄 Resume Screening App")

    job_desc = st.text_area("Paste Job Description")
    resume_folder = st.text_input("Enter Resume Folder Path", "resumes")

    if st.button("Process Resumes"):
        resume_info = process_resumes(resume_folder)
        if not resume_info:
            st.warning("⚠️ No resumes processed.")
            return

        df = pd.DataFrame(resume_info)

        # ✅ Load Model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Job embedding
        job_embedding = model.encode(job_desc, convert_to_tensor=True)

        # Resume embeddings
        resume_embeddings = model.encode(df["RawText"].tolist(), convert_to_tensor=True)

        # Cosine similarity
        similarities = util.pytorch_cos_sim(job_embedding, resume_embeddings)[0]
        scores = similarities.cpu().numpy()

        df["Score"] = scores
        df_sorted = df.sort_values(by="Score", ascending=False)

        st.subheader("🏆 Top Candidates")
        for i, row in df_sorted.head(10).iterrows():
            st.markdown(f"""
            **#{i+1} {row['Name']} {row['Score']:.3f}**  
            📧 Email: {row['Email']}  
            🛠 Skills: {', '.join(row['Skills']) if row['Skills'] else 'Not Found'}  
            """)

        # Save results
        csv_path = "/content/drive/MyDrive/resume_extracted_data.csv"
        df_sorted.to_csv(csv_path, index=False)
        st.success(f"✅ Data saved to {csv_path}")

if __name__ == "__main__":
    main()
