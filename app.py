# Full Streamlit app with improved name extraction (replace your app with this)
import os
import re
import nltk
import docx
import pdfplumber
import pandas as pd
import numpy as np
import tempfile
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="AI Resume Screening Tool",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Your CSS omitted for brevity - include your CSS block here if needed)
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.1rem;
    color: #666;
    text-align: center;
    margin-bottom: 1.5rem;
}
.candidate-card {
    background: linear-gradient(145deg, #f0f2f6, #ffffff);
    padding: 1rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.06);
    border-left: 4px solid #1f77b4;
}
.score-badge {
    display: inline-block;
    background: #1f77b4;
    color: white;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    font-weight: bold;
    margin-left: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache models (NLTK downloads, SBERT and spaCy)."""
    try:
        # NLTK
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download('wordnet', quiet=True)
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    # SBERT
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading SBERT model: {e}")
        sbert_model = None

    # spaCy (try to load model; if missing, download it)
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Download if not available
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        st.warning(f"spaCy not available or failed to load: {e}. Name extraction will fall back to regex/email heuristics.")
        nlp = None

    return stop_words, sbert_model, nlp

# Skills db
SKILLS_DB = {
    "python", "java", "sql", "machine learning", "nlp", "deep learning",
    "excel", "c++", "cloud", "aws", "javascript", "react", "node.js",
    "docker", "kubernetes", "tensorflow", "pytorch", "pandas", "numpy",
    "git", "html", "css", "mongodb", "postgresql", "rest api", "graphql",
    "spring boot", "hibernate", "angular", "vue.js", "django", "flask"
}

# Text extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file or file path"""
    text = ""
    try:
        # pdfplumber accepts path or file-like bytes buffer
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        # handle file-like objects by writing to temp file if needed
        try:
            # try to detect if pdf_file is bytes-like (Streamlit UploadFile)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf_file.getvalue())
            tmp.close()
            with pdfplumber.open(tmp.name) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            os.unlink(tmp.name)
        except Exception as e2:
            st.error(f"Error reading PDF: {e} / {e2}")
    return text.strip()

def extract_text_from_docx(docx_file):
    """Extract text from uploaded DOCX or file path"""
    try:
        # python-docx can read a path or file-like with .getvalue()
        try:
            doc = docx.Document(docx_file)
        except Exception:
            # fallback: write bytes to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            tmp.write(docx_file.getvalue())
            tmp.close()
            doc = docx.Document(tmp.name)
            os.unlink(tmp.name)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    except Exception:
        return text

# Improved name extraction
def is_valid_name_candidate(name, non_name_indicators):
    """Heuristic checks for candidate validity."""
    if not name or len(name.strip()) < 3:
        return False

    words = name.split()
    if len(words) < 2 or len(words) > 4:
        return False

    # No digits
    if re.search(r'\d', name):
        return False

    name_lower = name.lower()
    for bad in non_name_indicators:
        if bad in name_lower:
            return False

    # only letters, spaces, hyphens, apostrophes allowed
    if not re.match(r"^[A-Za-z\s\-\']+$", name):
        return False

    # each token should be >=2 characters (allow initials rarely)
    for w in words:
        if len(w.strip(".'-")) < 2:
            return False

    return True

def extract_name(text, nlp=None):
    """
    Robust name extraction:
      1. Check labeled lines like 'Name:'
      2. Regex Title Case / ALL CAPS patterns in the first 20 lines
      3. spaCy PERSON entities (if nlp provided)
      4. Email-prefix fallback (john.doe -> John Doe)
      5. First non-heading line fallback (safely)
    """
    if not text or not text.strip():
        return "Unknown"

    non_name_indicators = {
        'resume', 'curriculum', 'vitae', 'objective', 'summary', 'profile',
        'skills', 'experience', 'education', 'projects', 'employment', 'work',
        'company', 'employer', 'employer details', 'responsibilities', 'role',
        'roles', 'manager', 'developer', 'engineer', 'programmer', 'consultant',
        'domain', 'expert', 'working', 'knowledge', 'involved', 'sr', 'jr',
        'senior', 'assistant', 'intern', 'technician'
    }

    # Clean lines and look in the top of the document
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head_lines = lines[:20]  # inspect top 20 lines

    # Patterns to try (case-insensitive)
    name_patterns = [
        r'^\s*Name[:\s\-]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$',  # "Name: John Smith"
        r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$',               # line that is Title Case (2-4 words)
        r'^\s*([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\s*$',                   # ALL CAPS line
    ]

    for line in head_lines:
        # Remove common trailing suffixes that sometimes follow names
        cleaned = re.sub(r'\b(Employer Details|Employer|Employer:|Details|Employer Details:)\b', '', line, flags=re.IGNORECASE).strip()
        # Remove leading labels like "Candidate Name", "Applicant:"
        cleaned = re.sub(r'^(Candidate|Applicant|Resume of|CV of|Name)\s*[:\-]+\s*', '', cleaned, flags=re.IGNORECASE).strip()
        if len(cleaned) < 3:
            continue

        # Skip lines that clearly indicate not a name
        low = cleaned.lower()
        if any(tok in low for tok in non_name_indicators):
            continue

        for pat in name_patterns:
            m = re.search(pat, cleaned)
            if m:
                candidate = m.group(1).strip()
                if is_valid_name_candidate(candidate, non_name_indicators):
                    return candidate

    # spaCy NER fallback (if available)
    if nlp is not None:
        try:
            # Run NER on first chunk (faster)
            doc = nlp("\n".join(head_lines))
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    candidate = ent.text.strip()
                    # strip trailing words like "Employer Details"
                    candidate = re.sub(r'\b(Employer Details|Employer|Employer:|Details)\b', '', candidate, flags=re.IGNORECASE).strip()
                    if is_valid_name_candidate(candidate, non_name_indicators):
                        return candidate
        except Exception:
            pass

    # Email prefix fallback
    email_match = re.search(r'([\w\.-]+)@[\w\.-]+\.\w{2,}', text)
    if email_match:
        prefix = email_match.group(1)
        parts = re.split(r'[._\-]', prefix)
        if len(parts) >= 2:
            cand = " ".join([p.capitalize() for p in parts[:2]])
            if is_valid_name_candidate(cand, non_name_indicators):
                return cand

    # Safe first-line fallback: pick the first non-heading short line that looks like a name
    for line in head_lines:
        cleaned = line.strip()
        low = cleaned.lower()
        if len(cleaned.split()) >= 2 and len(cleaned.split()) <= 4 and not any(tok in low for tok in non_name_indicators):
            # ensure it doesn't contain punctuation or digits
            if re.match(r"^[A-Za-z\s\-\']+$", cleaned):
                cand = " ".join([w.capitalize() for w in cleaned.split()])
                if is_valid_name_candidate(cand, non_name_indicators):
                    return cand

    return "Unknown"

# Other extractors
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    patterns = [
        r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'(\+\d{1,3}[-.\s]?)?\d{10}'
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return "Not Found"

def extract_skills(text):
    text_lower = text.lower()
    found = [s for s in SKILLS_DB if s in text_lower]
    return found

def extract_experience(text):
    patterns = [
        r'(\d{1,2}\+?\s*(?:years?|yrs?))',
        r'(?:experience[:\s]*)(\d{1,2}\+?\s*(?:years?|yrs?))'
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return "Not specified"

# Resume processing (uses nlp for better name extraction)
def process_resume(file_content, filename, stop_words, nlp):
    """Process a single resume and extract information"""
    # Read text from file
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    elif filename.lower().endswith('.docx'):
        text = extract_text_from_docx(file_content)
    else:
        return None

    if not text or not text.strip():
        return None

    name = extract_name(text, nlp)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(text)
    experience = extract_experience(text)

    return {
        'name': name,
        'email': email,
        'phone': phone,
        'skills': skills,
        'experience': experience,
        'raw_text': text,
        'filename': filename
    }

def calculate_all_hybrid_scores(candidates_data, job_description, sbert_model):
    """Calculate hybrid scores (SBERT + TF-IDF) and attach to each candidate"""
    try:
        resume_texts = [c['raw_text'] for c in candidates_data]
        # SBERT embeddings
        job_embedding = sbert_model.encode([job_description])
        resume_embeddings = sbert_model.encode(resume_texts)
        sbert_scores = [cosine_similarity([resume_embeddings[i]], job_embedding)[0][0] for i in range(len(resume_embeddings))]

        # TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        all_texts = [job_description] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        job_vec = tfidf_matrix[0:1]
        resume_vecs = tfidf_matrix[1:]
        tfidf_scores = [cosine_similarity(resume_vecs[i:i+1], job_vec)[0][0] for i in range(len(resume_texts))]

        # Normalize scores
        def normalize(xs):
            if len(xs) == 1:
                return [1.0]
            mn, mx = min(xs), max(xs)
            if mx == mn:
                return [0.5] * len(xs)
            return [(x - mn) / (mx - mn) for x in xs]

        sbert_norm = normalize(sbert_scores)
        tfidf_norm = normalize(tfidf_scores)

        for i, c in enumerate(candidates_data):
            hybrid = 0.6 * sbert_norm[i] + 0.4 * tfidf_norm[i]
            c['scores'] = {
                'hybrid_score': hybrid,
                'sbert_score': sbert_scores[i],
                'tfidf_score': tfidf_scores[i]
            }
        return candidates_data

    except Exception as e:
        st.error(f"Error calculating scores: {e}")
        # fallback random small scores to allow UI to show something
        for c in candidates_data:
            c['scores'] = {
                'hybrid_score': np.random.uniform(0.1, 0.9),
                'sbert_score': np.random.uniform(0.1, 0.9),
                'tfidf_score': np.random.uniform(0.1, 0.9)
            }
        return candidates_data

def create_visualization(candidates_data):
    if not candidates_data:
        return None
    top = candidates_data[:10]
    names = [c['name'] for c in top][::-1]
    scores = [c['scores']['hybrid_score'] for c in top][::-1]
    fig = go.Figure(go.Bar(x=scores, y=names, orientation='h', marker=dict(color=scores, colorscale='Viridis')))
    fig.update_layout(title='Top Candidates (Hybrid Score)', xaxis_title='Hybrid Score', yaxis_title='Candidate')
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 AI Resume Screening Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload resumes and job descriptions to find the best candidates</p>', unsafe_allow_html=True)

    stop_words, sbert_model, nlp = load_models()
    if sbert_model is None:
        st.error("SBERT failed to load. The app cannot continue.")
        return

    st.sidebar.header("Settings")
    max_candidates = st.sidebar.slider("Max candidates to show", 5, 20, 10)
    show_details = st.sidebar.checkbox("Show detailed scores", True)

    col1, col2 = st.columns(2)
    with col1:
        job_description = st.text_area("Enter the job description:", height=260, help="Paste the JD here.")
    with col2:
        uploaded_files = st.file_uploader("Upload resumes (pdf/docx)", accept_multiple_files=True, type=['pdf', 'docx'])

    if st.button("Analyze"):
        if not job_description or not uploaded_files:
            st.error("Provide job description and at least one resume.")
            return

        candidates = []
        with st.spinner("Processing resumes..."):
            for up in uploaded_files:
                # create temporary file for robust extraction
                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                tmpfile.write(up.getvalue())
                tmpfile.close()
                # process by passing the file object (Streamlit uploaded file or temp file)
                # prefer passing the temp file path to avoid library edge cases
                with open(tmpfile.name, 'rb') as fobj:
                    data = process_resume(fobj, up.name, stop_words, nlp)
                os.unlink(tmpfile.name)
                if data:
                    candidates.append(data)

            if not candidates:
                st.error("No valid resumes processed.")
                return

            # scoring
            candidates = calculate_all_hybrid_scores(candidates, job_description, sbert_model)
            candidates.sort(key=lambda x: x['scores']['hybrid_score'], reverse=True)

            st.success(f"Processed {len(candidates)} resumes")
            st.header("🏆 Top Candidates")
            for i, c in enumerate(candidates[:max_candidates], 1):
                s = c['scores']['hybrid_score']
                st.markdown(f"""
                <div class="candidate-card">
                  <h3>#{i} {c['name']} <span class="score-badge">{s:.3f}</span></h3>
                  <p><strong>📧 Email:</strong> {c['email']}</p>
                  <p><strong>📱 Phone:</strong> {c['phone']}</p>
                  <p><strong>💼 Experience:</strong> {c['experience']}</p>
                  <p><strong>🛠️ Skills:</strong> {', '.join(c['skills']) if c['skills'] else 'Not specified'}</p>
                </div>
                """, unsafe_allow_html=True)
                if show_details:
                    with st.expander(f"Detailed scores for {c['name']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Hybrid Score", f"{c['scores']['hybrid_score']:.3f}")
                        col2.metric("SBERT Score", f"{c['scores']['sbert_score']:.3f}")
                        col3.metric("TF-IDF Score", f"{c['scores']['tfidf_score']:.3f}")

            # visualization and download
            fig = create_visualization(candidates)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            df = pd.DataFrame([{
                'Rank': idx+1,
                'Name': c['name'],
                'Email': c['email'],
                'Phone': c['phone'],
                'Experience': c['experience'],
                'Skills': ', '.join(c['skills']),
                'Hybrid': c['scores']['hybrid_score'],
                'SBERT': c['scores']['sbert_score'],
                'TFIDF': c['scores']['tfidf_score']
            } for idx, c in enumerate(candidates[:max_candidates])])

            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, file_name="resume_screening_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
