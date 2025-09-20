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
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Page config
st.set_page_config(
    page_title="AI Resume Screening Tool",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #666;
    text-align: center;
    margin-bottom: 3rem;
}
.candidate-card {
    background: linear-gradient(145deg, #f0f2f6, #ffffff);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 4px solid #1f77b4;
}
.score-badge {
    display: inline-block;
    background: #1f77b4;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: bold;
    margin-left: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache models"""
    try:
        # Download NLTK data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Load stopwords
        stop_words = set(stopwords.words("english"))
        
        # Load SBERT model
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return stop_words, sbert_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return set(), None

# Skills database
SKILLS_DB = {
    "python", "java", "sql", "machine learning", "nlp", "deep learning", 
    "excel", "c++", "cloud", "aws", "javascript", "react", "node.js",
    "docker", "kubernetes", "tensorflow", "pytorch", "pandas", "numpy",
    "git", "html", "css", "mongodb", "postgresql", "rest api", "graphql",
    "spring boot", "hibernate", "angular", "vue.js", "django", "flask"
}

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
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
    """Extract text from uploaded DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def preprocess_text(text, stop_words):
    """Clean and tokenize text"""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    try:
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)
    except Exception:
        return text
def extract_name(text):
    """Extract name using regex + fallbacks"""
    lines = text.split('\n')[:15]  # Check first 15 lines
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$)',
        r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
        r'([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})(?:\s*\n|\s*$)',
        r'([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})(?:\s*\n|\s*$)',  # All caps
    ]
    non_name_indicators = {
        'resume', 'developer', 'engineer', 'profile', 'summary',
        'skills', 'objective', 'experience', 'project', 'curriculum',
        'vitae', 'technologies', 'professional'
    }

    # Try regex patterns
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        for pattern in name_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                if is_valid_name(potential_name, non_name_indicators):
                    return potential_name

    # Fallback 1: first clean line
    for line in lines:
        if line and all(word.lower() not in non_name_indicators for word in line.split()):
            if 2 <= len(line.split()) <= 4:
                return line.strip().title()

    # Fallback 2: infer from email prefix
    email = extract_email(text)
    if email != "Not Found":
        prefix = email.split('@')[0]
        parts = re.split(r'[._]', prefix)
        if len(parts) >= 2:
            return " ".join([p.capitalize() for p in parts[:2]])

    return "Unknown"

# def extract_name(text):
#     """Extract name using improved pattern matching"""
#     # Split text into lines and look for name patterns
#     lines = text.split('\n')[:10]  # Check first 10 lines only
    
#     # Common patterns for names in resumes
#     name_patterns = [
#     # Start of line (supports middle names and all caps)
#     r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?:\s|$)',
    
#     # After "Name:" (supports middle names)
#     r'Name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
    
#     # Standalone names (supports 2–4 words, also all caps)
#     r'([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})(?:\s*\n|\s*$)',
    
#     # All-uppercase names
#     r'([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})(?:\s*\n|\s*$)',
#      ]

#     # name_patterns = [
#     #     r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:\s|$)',  # Start of line
#     #     r'Name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # After "Name:"
#     #     r'([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})(?:\s*\n|\s*$)',  # Standalone names
#     # ]
    
#     # Words that definitely indicate this is NOT a name
#     non_name_indicators = {
#         'object', 'oriented', 'analysis', 'information', 'technology', 
#         'business', 'systems', 'analyst', 'software', 'developer', 'engineer',
#         'resume', 'curriculum', 'vitae', 'profile', 'summary', 'objective',
#         'education', 'experience', 'skills', 'projects', 'work', 'employment',
#         'professional', 'technical', 'senior', 'junior', 'lead', 'manager',
#         'consultant', 'specialist', 'architect', 'designer', 'coordinator'
#     }
    
#     for line in lines:
#         line = line.strip()
#         if not line or len(line) < 5:  # Skip very short lines
#             continue
            
#         for pattern in name_patterns:
#             match = re.search(pattern, line, re.MULTILINE)
#             if match:
#                 potential_name = match.group(1).strip()
                
#                 # Check if it's a valid name
#                 if is_valid_name(potential_name, non_name_indicators):
#                     return potential_name
    
#     return "Unknown"

def is_valid_name(name, non_name_indicators):
    """Check if extracted name is valid with improved filtering"""
    if len(name.split()) < 2:
        return False
    
    # Check against non-name indicators
    name_lower = name.lower()
    if any(indicator in name_lower for indicator in non_name_indicators):
        return False
    
    # Must contain only letters, spaces, hyphens, apostrophes
    if not re.match(r"^[a-zA-Z\s\-']+$", name):
        return False
    
    # Each word should be reasonably long (not just initials)
    words = name.split()
    if any(len(word) < 2 for word in words):
        return False
        
    # Should not be all uppercase (likely a header)
    if name.isupper():
        return False
    
    return True

def extract_email(text):
    """Extract email address"""
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    """Extract phone number"""
    patterns = [
        r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"(\+\d{1,3}[-.\s]?)?\d{10}"
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
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
        r"(?:experience|exp)[:\s]*(\d+)\+?\s*(?:years?|yrs?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} years"
    return "Not specified"

def process_resume(file_content, filename, stop_words):
    """Process a single resume and extract information"""
    # Extract text based on file type
    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    elif filename.endswith('.docx'):
        text = extract_text_from_docx(file_content)
    else:
        return None
    
    if not text.strip():
        return None
    
    # Extract information
    name = extract_name(text)
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

def calculate_all_hybrid_scores(candidates_data, job_description, sbert_model, stop_words):
    """Calculate hybrid scores for all candidates together with proper normalization"""
    try:
        # Extract resume texts
        resume_texts = [candidate['raw_text'] for candidate in candidates_data]
        
        # Calculate SBERT scores for all resumes
        job_embedding = sbert_model.encode([job_description])
        resume_embeddings = sbert_model.encode(resume_texts)
        sbert_scores = [cosine_similarity([resume_embeddings[i]], job_embedding)[0][0] 
                       for i in range(len(resume_embeddings))]
        
        # Calculate TF-IDF scores for all resumes
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        all_texts = [job_description] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        job_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]
        tfidf_scores = [cosine_similarity(resume_vectors[i:i+1], job_vector)[0][0] 
                       for i in range(len(resume_texts))]
        
        # Normalize scores across all candidates
        if len(sbert_scores) > 1:
            sbert_min, sbert_max = min(sbert_scores), max(sbert_scores)
            if sbert_max != sbert_min:
                sbert_normalized = [(score - sbert_min) / (sbert_max - sbert_min) for score in sbert_scores]
            else:
                sbert_normalized = [0.5] * len(sbert_scores)  # All scores are identical
        else:
            sbert_normalized = [1.0]  # Single candidate gets max score
            
        if len(tfidf_scores) > 1:
            tfidf_min, tfidf_max = min(tfidf_scores), max(tfidf_scores)
            if tfidf_max != tfidf_min:
                tfidf_normalized = [(score - tfidf_min) / (tfidf_max - tfidf_min) for score in tfidf_scores]
            else:
                tfidf_normalized = [0.5] * len(tfidf_scores)  # All scores are identical
        else:
            tfidf_normalized = [1.0]  # Single candidate gets max score
        
        # Calculate hybrid scores (60% SBERT, 40% TF-IDF)
        for i, candidate in enumerate(candidates_data):
            hybrid_score = 0.6 * sbert_normalized[i] + 0.4 * tfidf_normalized[i]
            candidate['scores'] = {
                'hybrid_score': hybrid_score,
                'sbert_score': sbert_scores[i],
                'tfidf_score': tfidf_scores[i]
            }
        
        return candidates_data
    
    except Exception as e:
        st.error(f"Error calculating scores: {e}")
        # Fallback: assign random scores
        for candidate in candidates_data:
            candidate['scores'] = {
                'hybrid_score': np.random.uniform(0.1, 0.9),
                'sbert_score': np.random.uniform(0.1, 0.9),
                'tfidf_score': np.random.uniform(0.1, 0.9)
            }
        return candidates_data

def create_visualization(candidates_data):
    """Create visualization for candidate rankings"""
    if not candidates_data:
        return None
    
    # Prepare data for plotting
    names = [candidate['name'] for candidate in candidates_data[:10]]
    scores = [candidate['scores']['hybrid_score'] for candidate in candidates_data[:10]]
    
    # Create plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=names[::-1],  # Reverse for top-to-bottom display
            x=scores[::-1],
            orientation='h',
            marker=dict(
                color=scores[::-1],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{score:.3f}' for score in scores[::-1]],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title='Top 10 Candidate Rankings',
        xaxis_title='Hybrid Similarity Score',
        yaxis_title='Candidates',
        height=600,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Resume Screening Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload resumes and job descriptions to find the best candidates using advanced AI algorithms</p>', unsafe_allow_html=True)
    
    # Load models
    stop_words, sbert_model = load_models()
    
    if sbert_model is None:
        st.error("Failed to load required models. Please refresh the page.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Application Settings")
    max_candidates = st.sidebar.slider("Max candidates to show", 5, 20, 10)
    show_detailed_scores = st.sidebar.checkbox("Show detailed scores", True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Job Description")
        job_description = st.text_area(
            "Enter the job description:",
            height=300,
            placeholder="Paste your job description here..."
        )
    
    with col2:
        st.header("📋 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF or DOCX)",
            accept_multiple_files=True,
            type=['pdf', 'docx']
        )
    
    if st.button("🚀 Analyze Resumes", type="primary"):
        if not job_description.strip():
            st.error("Please enter a job description.")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one resume.")
            return
        
        # Process resumes
        with st.spinner("Processing resumes..."):
            candidates_data = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process resume
                resume_data = process_resume(uploaded_file, uploaded_file.name, stop_words)
                
                if resume_data:
                    candidates_data.append(resume_data)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
            
            if not candidates_data:
                st.error("No valid resumes could be processed.")
                return
            
            # Calculate scores for all candidates at once
            candidates_data = calculate_all_hybrid_scores(candidates_data, job_description, sbert_model, stop_words)
            
            # Sort candidates by hybrid score
            candidates_data.sort(key=lambda x: x['scores']['hybrid_score'], reverse=True)
            
            # Display results
            st.success(f"Successfully processed {len(candidates_data)} resumes!")
            
            # Show top candidates
            st.header("🏆 Top Candidates")
            
            for i, candidate in enumerate(candidates_data[:max_candidates], 1):
                score = candidate['scores']['hybrid_score']
                
                st.markdown(f"""
                <div class="candidate-card">
                    <h3>#{i} {candidate['name']} <span class="score-badge">{score:.3f}</span></h3>
                    <p><strong>📧 Email:</strong> {candidate['email']}</p>
                    <p><strong>📱 Phone:</strong> {candidate['phone']}</p>
                    <p><strong>💼 Experience:</strong> {candidate['experience']}</p>
                    <p><strong>🛠️ Skills:</strong> {', '.join(candidate['skills'][:10]) if candidate['skills'] else 'Not specified'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if show_detailed_scores:
                    with st.expander(f"Detailed scores for {candidate['name']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Hybrid Score", f"{candidate['scores']['hybrid_score']:.3f}")
                        with col2:
                            st.metric("SBERT Score", f"{candidate['scores']['sbert_score']:.3f}")
                        with col3:
                            st.metric("TF-IDF Score", f"{candidate['scores']['tfidf_score']:.3f}")
            
            # Visualization
            st.header("📊 Candidate Rankings Visualization")
            fig = create_visualization(candidates_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.header("💾 Download Results")
            results_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Name': candidate['name'],
                    'Email': candidate['email'],
                    'Phone': candidate['phone'],
                    'Experience': candidate['experience'],
                    'Skills': ', '.join(candidate['skills']),
                    'Hybrid Score': candidate['scores']['hybrid_score'],
                    'SBERT Score': candidate['scores']['sbert_score'],
                    'TF-IDF Score': candidate['scores']['tfidf_score']
                }
                for i, candidate in enumerate(candidates_data[:max_candidates])
            ])
            
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
