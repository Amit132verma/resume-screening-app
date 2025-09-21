import streamlit as st
import os
import re
import nltk
import docx
import pdfplumber
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import spacy

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize NLTK and SpaCy
@st.cache_resource
def initialize_nlp():
    """Initialize NLP resources"""
    # NLTK downloads
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # Load SpaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    return nlp

# Load models
@st.cache_resource
def load_models():
    """Load ML models"""
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model

# Skills database
SKILLS_DB = {
    "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin", "go",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "keras", "pandas", "numpy", "matplotlib",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "ci/cd",
    "react", "angular", "vue", "nodejs", "django", "flask", "spring", "express",
    "html", "css", "bootstrap", "tailwind", "sass",
    "agile", "scrum", "jira", "confluence",
    "excel", "powerpoint", "tableau", "power bi",
    "communication", "leadership", "teamwork", "problem solving"
}

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
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    except Exception as e:
        return text

def extract_name(text, nlp):
    """Extract name using SpaCy NER"""
    doc = nlp(text[:1000])  # Process first 1000 chars for speed
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Filter out common false positives
            if len(name.split()) >= 2 and not any(tech in name.lower() for tech in ['java', 'python', 'developer']):
                return name
    
    # Fallback: regex pattern
    match = re.search(r"([A-Z][a-z]+)\s+([A-Z][a-z]+)", text)
    return match.group(0) if match else "Unknown"

def extract_email(text):
    """Extract email address"""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    """Extract phone number"""
    patterns = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        r"\b\d{10}\b",
        r"\+\d{1,3}\s?\d{10}"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return "Not Found"

def extract_skills(text):
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in SKILLS_DB:
        if skill in text_lower:
            found_skills.append(skill)
    
    return found_skills

def extract_experience(text):
    """Extract years of experience"""
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'experience\s*:?\s*(\d+)\+?\s*(?:years?|yrs?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} years"
    return "Not Found"

def process_resume(file, nlp):
    """Process a single resume file"""
    # Extract text based on file type
    if file.name.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        return None
    
    if not text:
        return None
    
    # Extract information
    cleaned_text = preprocess_text(text)
    
    resume_data = {
        "filename": file.name,
        "name": extract_name(text, nlp),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "experience": extract_experience(text),
        "skills": extract_skills(cleaned_text),
        "raw_text": text,
        "cleaned_text": cleaned_text
    }
    
    return resume_data

def calculate_similarity(resume_data, job_description, sbert_model):
    """Calculate similarity scores using hybrid approach"""
    # Preprocess job description
    cleaned_job = preprocess_text(job_description)
    
    # SBERT embeddings
    job_embedding = sbert_model.encode([cleaned_job])
    resume_texts = [r['cleaned_text'] for r in resume_data]
    resume_embeddings = sbert_model.encode(resume_texts)
    
    # Calculate SBERT similarities
    sbert_scores = cosine_similarity(job_embedding, resume_embeddings)[0]
    
    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    all_texts = resume_texts + [cleaned_job]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate TF-IDF similarities
    tfidf_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    # Normalize scores
    scaler = MinMaxScaler()
    sbert_norm = scaler.fit_transform(sbert_scores.reshape(-1, 1)).flatten()
    tfidf_norm = scaler.fit_transform(tfidf_scores.reshape(-1, 1)).flatten()
    
    # Hybrid score (60% SBERT, 40% TF-IDF)
    hybrid_scores = 0.6 * sbert_norm + 0.4 * tfidf_norm
    
    # Add scores to resume data
    for i, resume in enumerate(resume_data):
        resume['sbert_score'] = float(sbert_scores[i])
        resume['tfidf_score'] = float(tfidf_scores[i])
        resume['hybrid_score'] = float(hybrid_scores[i])
    
    return resume_data

def main():
    st.title("🎯 AI-Powered Resume Screening System")
    st.markdown("### Upload resumes and enter job description to find the best candidates")
    
    # Initialize resources
    nlp = initialize_nlp()
    sbert_model = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Select Ranking Model",
            ["Hybrid (SBERT + TF-IDF)", "SBERT Only", "TF-IDF Only"]
        )
        
        # Top N candidates
        top_n = st.slider("Number of Top Candidates", 1, 20, 5)
        
        # Skill filter
        st.subheader("🔧 Skill Filters")
        required_skills = st.multiselect(
            "Required Skills",
            options=sorted(list(SKILLS_DB)),
            default=[]
        )
        
        st.markdown("---")
        st.info("💡 The hybrid model combines semantic understanding (SBERT) with keyword matching (TF-IDF) for best results.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple PDF or DOCX files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")
    
    with col2:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Enter the job description",
            height=200,
            placeholder="Paste the job description here..."
        )
    
    # Process button
    if st.button("🚀 Start Screening", type="primary"):
        if uploaded_files and job_description:
            with st.spinner("Processing resumes..."):
                # Process all resumes
                resume_data = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    data = process_resume(file, nlp)
                    if data:
                        resume_data.append(data)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                if not resume_data:
                    st.error("No valid resumes found!")
                    return
                
                # Calculate similarities
                resume_data = calculate_similarity(resume_data, job_description, sbert_model)
                
                # Apply skill filters if specified
                if required_skills:
                    filtered_data = []
                    for resume in resume_data:
                        if any(skill in resume['skills'] for skill in required_skills):
                            filtered_data.append(resume)
                    
                    if filtered_data:
                        resume_data = filtered_data
                    else:
                        st.warning("No resumes match the required skills. Showing all results.")
                
                # Sort by selected model
                if model_type == "SBERT Only":
                    resume_data.sort(key=lambda x: x['sbert_score'], reverse=True)
                    score_key = 'sbert_score'
                elif model_type == "TF-IDF Only":
                    resume_data.sort(key=lambda x: x['tfidf_score'], reverse=True)
                    score_key = 'tfidf_score'
                else:
                    resume_data.sort(key=lambda x: x['hybrid_score'], reverse=True)
                    score_key = 'hybrid_score'
                
                # Display results
                st.markdown("---")
                st.header("🏆 Top Candidates")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Resumes", len(resume_data))
                with col2:
                    st.metric("Avg Match Score", f"{np.mean([r[score_key] for r in resume_data]):.2%}")
                with col3:
                    st.metric("Top Score", f"{resume_data[0][score_key]:.2%}")
                with col4:
                    st.metric("Skills Found", len(set(sum([r['skills'] for r in resume_data], []))))
                
                # Top candidates
                st.subheader(f"Top {min(top_n, len(resume_data))} Candidates")
                
                for idx, resume in enumerate(resume_data[:top_n]):
                    with st.expander(f"#{idx+1} - {resume['name']} (Score: {resume[score_key]:.2%})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Email:** {resume['email']}")
                            st.write(f"**Phone:** {resume['phone']}")
                            st.write(f"**Experience:** {resume['experience']}")
                            st.write(f"**Skills:** {', '.join(resume['skills'][:10])}")
                        
                        with col2:
                            st.write("**Similarity Scores:**")
                            st.write(f"- SBERT: {resume['sbert_score']:.2%}")
                            st.write(f"- TF-IDF: {resume['tfidf_score']:.2%}")
                            st.write(f"- Hybrid: {resume['hybrid_score']:.2%}")
                
                # Visualizations
                st.markdown("---")
                st.header("📊 Analytics Dashboard")
                
                tab1, tab2, tab3 = st.tabs(["Score Distribution", "Skills Analysis", "Comparison"])
                
                with tab1:
                    # Score distribution chart
                    fig = go.Figure()
                    
                    top_candidates = resume_data[:top_n]
                    names = [r['name'][:20] for r in top_candidates]
                    
                    fig.add_trace(go.Bar(
                        name='SBERT',
                        x=names,
                        y=[r['sbert_score'] for r in top_candidates],
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='TF-IDF',
                        x=names,
                        y=[r['tfidf_score'] for r in top_candidates],
                        marker_color='lightgreen'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Hybrid',
                        x=names,
                        y=[r['hybrid_score'] for r in top_candidates],
                        marker_color='coral'
                    ))
                    
                    fig.update_layout(
                        title="Score Comparison",
                        xaxis_title="Candidates",
                        yaxis_title="Score",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Skills frequency
                    all_skills = []
                    for resume in resume_data[:top_n]:
                        all_skills.extend(resume['skills'])
                    
                    skill_counts = pd.Series(all_skills).value_counts().head(15)
                    
                    fig = px.bar(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        orientation='h',
                        title="Most Common Skills",
                        labels={'x': 'Frequency', 'y': 'Skills'},
                        color=skill_counts.values,
                        color_continuous_scale='viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Create comparison table
                    comparison_data = []
                    for resume in resume_data[:top_n]:
                        comparison_data.append({
                            'Name': resume['name'],
                            'Email': resume['email'],
                            'Experience': resume['experience'],
                            'Skills Count': len(resume['skills']),
                            'Match Score': f"{resume[score_key]:.2%}"
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button for results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="screening_results.csv",
                        mime="text/csv"
                    )
        else:
            if not uploaded_files:
                st.warning("Please upload resume files")
            if not job_description:
                st.warning("Please enter a job description")

if __name__ == "__main__":
    main()
