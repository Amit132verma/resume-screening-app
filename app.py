import streamlit as st
import os
import re
import nltk
import docx
import pdfplumber
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import tempfile
import shutil

# Page configuration
st.set_page_config(
    page_title="Resume Screening Tool",
    page_icon="📄",
    layout="wide"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception as e:
        st.error(f"Error loading NLTK data: {e}")
        return set()

# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize
stop_words = download_nltk_data()
model = load_model()
skills_db = {"python", "java", "sql", "machine learning", "nlp", "deep learning", 
             "excel", "c++", "cloud", "aws", "react", "nodejs", "docker", "kubernetes"}

# Text extraction functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text):
    """Clean and tokenize text"""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        return text

def extract_name(text):
    """Extract name using regex"""
    match = re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", text)
    return match.group(0) if match else "Unknown"

def extract_email(text):
    """Extract email address"""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_skills(text):
    """Extract skills from text"""
    words = set(text.lower().split())
    return list(skills_db.intersection(words))

def process_resume(file):
    """Process single resume file"""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(file)
    else:
        return None
    
    if not text:
        return None
    
    cleaned_text = preprocess_text(text)
    
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Skills": extract_skills(cleaned_text),
        "RawText": text,
        "FileName": file.name
    }

def rank_candidates(job_description, resumes):
    """Rank candidates based on job description"""
    if not resumes:
        return []
    
    cleaned_job_desc = preprocess_text(job_description)
    job_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)
    
    ranked_resumes = []
    for resume in resumes:
        resume_embedding = model.encode(resume["RawText"], convert_to_tensor=True)
        cosine_sim = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0].item()
        
        ranked_resumes.append({
            "Name": resume["Name"],
            "Email": resume["Email"],
            "Skills": resume["Skills"],
            "FileName": resume["FileName"],
            "SimilarityScore": cosine_sim
        })
    
    return sorted(ranked_resumes, key=lambda x: x["SimilarityScore"], reverse=True)

# Streamlit UI
st.title("📄 AI-Powered Resume Screening Tool")
st.markdown("Upload resumes and enter a job description to find the best matching candidates")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    num_candidates = st.slider("Number of top candidates to display", 1, 10, 5)
    st.markdown("---")
    st.markdown("### About")
    st.info("This tool uses SBERT (Sentence-BERT) to match resumes with job descriptions based on semantic similarity.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload one or more resume files"
    )

with col2:
    st.subheader("📝 Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="e.g., Looking for a Senior Python Developer with experience in Machine Learning, NLP, and Cloud technologies..."
    )

# Process button
if st.button("🔍 Screen Resumes", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one resume file")
    elif not job_description.strip():
        st.warning("Please enter a job description")
    else:
        with st.spinner("Processing resumes..."):
            # Process all resumes
            resumes = []
            for file in uploaded_files:
                resume_data = process_resume(file)
                if resume_data:
                    resumes.append(resume_data)
            
            if not resumes:
                st.error("No valid resumes could be processed")
            else:
                # Rank candidates
                ranked_candidates = rank_candidates(job_description, resumes)
                
                # Display results
                st.success(f"✅ Processed {len(resumes)} resumes successfully!")
                
                st.markdown("---")
                st.subheader(f"🏆 Top {num_candidates} Matching Candidates")
                
                # Display top candidates
                for i, candidate in enumerate(ranked_candidates[:num_candidates], 1):
                    with st.expander(f"#{i} - {candidate['Name']} (Score: {candidate['SimilarityScore']:.2%})"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Email:**", candidate['Email'])
                            st.write("**File:**", candidate['FileName'])
                        with col_b:
                            st.write("**Skills:**", ", ".join(candidate['Skills']) if candidate['Skills'] else "None detected")
                        
                        # Progress bar for similarity score
                        st.progress(candidate['SimilarityScore'])
                
                # Visualization
                st.markdown("---")
                st.subheader("📊 Candidate Ranking Visualization")
                
                top_n = ranked_candidates[:num_candidates]
                fig = go.Figure(go.Bar(
                    x=[c['SimilarityScore'] for c in top_n],
                    y=[c['Name'] for c in top_n],
                    orientation='h',
                    marker=dict(
                        color=[c['SimilarityScore'] for c in top_n],
                        colorscale='Blues',
                        showscale=True
                    ),
                    text=[f"{c['SimilarityScore']:.2%}" for c in top_n],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Candidate Similarity Scores",
                    xaxis_title="Similarity Score",
                    yaxis_title="Candidate",
                    height=400,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.markdown("---")
                df_results = pd.DataFrame([{
                    'Rank': i+1,
                    'Name': c['Name'],
                    'Email': c['Email'],
                    'Skills': ', '.join(c['Skills']),
                    'Similarity Score': f"{c['SimilarityScore']:.2%}",
                    'File': c['FileName']
                } for i, c in enumerate(top_n)])
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="resume_screening_results.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit & SBERT | Resume Screening Tool v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
