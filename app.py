import streamlit as st
import os
import re
import nltk
import docx
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Resume Screening Tool",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Download NLTK resources
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception as e:
        st.warning(f"NLTK download issue: {e}. Using basic preprocessing.")
        return set()

# Load SBERT model
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize
try:
    stop_words = download_nltk_data()
    model = load_model()
    st.session_state.initialized = True
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

skills_db = {
    "python", "java", "sql", "machine learning", "nlp", "deep learning", 
    "excel", "c++", "cloud", "aws", "react", "nodejs", "docker", "kubernetes",
    "javascript", "typescript", "spring", "hibernate", "django", "flask",
    "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "git",
    "agile", "scrum", "ci/cd", "linux", "windows", "azure", "gcp"
}

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
        st.error(f"Error reading PDF {pdf_file.name}: {str(e)}")
    return text.strip()

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX {docx_file.name}: {str(e)}")
        return ""

def preprocess_text(text):
    """Clean and tokenize text"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    
    if stop_words:
        try:
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            return " ".join(tokens)
        except:
            pass
    
    # Fallback if tokenization fails
    words = text.split()
    return " ".join([w for w in words if len(w) > 2])

def extract_name(text):
    """Extract name using regex"""
    lines = text.split('\n')
    # Check first few lines for name
    for line in lines[:5]:
        line = line.strip()
        # Look for capitalized words (likely a name)
        match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', line)
        if match:
            return match.group(1)
    return "Unknown"

def extract_email(text):
    """Extract email address"""
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    """Extract phone number"""
    match = re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    return match.group(0) if match else "Not Found"

def extract_skills(text):
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

def process_resume(file):
    """Process single resume file"""
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_extension in ['docx', 'doc']:
            text = extract_text_from_docx(file)
        else:
            st.warning(f"Unsupported file format: {file.name}")
            return None
        
        if not text or len(text) < 50:
            st.warning(f"Could not extract sufficient text from {file.name}")
            return None
        
        cleaned_text = preprocess_text(text)
        
        return {
            "Name": extract_name(text),
            "Email": extract_email(text),
            "Phone": extract_phone(text),
            "Skills": extract_skills(text),
            "RawText": text,
            "CleanedText": cleaned_text,
            "FileName": file.name
        }
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def rank_candidates(job_description, resumes):
    """Rank candidates based on job description"""
    if not resumes or not model:
        return []
    
    try:
        cleaned_job_desc = preprocess_text(job_description)
        job_embedding = model.encode(cleaned_job_desc, convert_to_tensor=True)
        
        ranked_resumes = []
        for resume in resumes:
            resume_text = resume["CleanedText"] if resume["CleanedText"] else resume["RawText"]
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(job_embedding, resume_embedding)[0][0].item()
            
            ranked_resumes.append({
                "Name": resume["Name"],
                "Email": resume["Email"],
                "Phone": resume["Phone"],
                "Skills": resume["Skills"],
                "FileName": resume["FileName"],
                "SimilarityScore": cosine_sim
            })
        
        return sorted(ranked_resumes, key=lambda x: x["SimilarityScore"], reverse=True)
    except Exception as e:
        st.error(f"Error ranking candidates: {str(e)}")
        return []

# Streamlit UI
st.title("📄 AI-Powered Resume Screening Tool")
st.markdown("Upload resumes and enter a job description to find the best matching candidates")

# Check if model loaded successfully
if not model:
    st.error("Failed to load the AI model. Please refresh the page.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    num_candidates = st.slider("Number of top candidates", 1, 10, 5)
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    if 'total_resumes' in st.session_state:
        st.metric("Resumes Processed", st.session_state.total_resumes)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("This tool uses SBERT for semantic similarity matching between resumes and job descriptions.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload one or more resume files (PDF or DOCX format)"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

with col2:
    st.subheader("📝 Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200,
        placeholder="Example: Looking for a Senior Python Developer with 5+ years experience in Machine Learning, NLP, and Cloud technologies (AWS/Azure). Strong knowledge of Deep Learning frameworks required.",
        help="Paste the complete job description here"
    )

# Process button
st.markdown("---")
if st.button("🔍 Screen Resumes", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume file")
    elif not job_description.strip():
        st.warning("⚠️ Please enter a job description")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process all resumes
        resumes = []
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            resume_data = process_resume(file)
            if resume_data:
                resumes.append(resume_data)
        
        progress_bar.empty()
        status_text.empty()
        
        if not resumes:
            st.error("❌ No valid resumes could be processed. Please check your files.")
        else:
            st.session_state.total_resumes = len(resumes)
            
            # Rank candidates
            with st.spinner("Ranking candidates..."):
                ranked_candidates = rank_candidates(job_description, resumes)
            
            if not ranked_candidates:
                st.error("❌ Error ranking candidates. Please try again.")
            else:
                # Display results
                st.success(f"✅ Successfully processed {len(resumes)} resume(s)!")
                
                st.markdown("---")
                st.subheader(f"🏆 Top {min(num_candidates, len(ranked_candidates))} Matching Candidates")
                
                # Display top candidates
                for i, candidate in enumerate(ranked_candidates[:num_candidates], 1):
                    score_percentage = candidate['SimilarityScore'] * 100
                    
                    # Color code based on score
                    if score_percentage >= 70:
                        color = "🟢"
                    elif score_percentage >= 50:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    with st.expander(f"{color} #{i} - {candidate['Name']} | Match: {score_percentage:.1f}%", expanded=(i<=3)):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**📧 Email:**", candidate['Email'])
                            st.write("**📱 Phone:**", candidate['Phone'])
                            st.write("**📄 File:**", candidate['FileName'])
                        
                        with col_b:
                            if candidate['Skills']:
                                st.write("**🔧 Skills Found:**")
                                skills_str = ", ".join(candidate['Skills'][:10])
                                if len(candidate['Skills']) > 10:
                                    skills_str += f" (+{len(candidate['Skills'])-10} more)"
                                st.write(skills_str)
                            else:
                                st.write("**🔧 Skills:** None detected")
                        
                        # Progress bar for similarity score
                        st.progress(candidate['SimilarityScore'])
                        st.caption(f"Similarity Score: {candidate['SimilarityScore']:.4f}")
                
                # Visualization
                if len(ranked_candidates) > 0:
                    st.markdown("---")
                    st.subheader("📊 Candidate Ranking Visualization")
                    
                    top_n = ranked_candidates[:num_candidates]
                    
                    fig = go.Figure(go.Bar(
                        x=[c['SimilarityScore'] * 100 for c in top_n],
                        y=[c['Name'] for c in top_n],
                        orientation='h',
                        marker=dict(
                            color=[c['SimilarityScore'] * 100 for c in top_n],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Match %")
                        ),
                        text=[f"{c['SimilarityScore']*100:.1f}%" for c in top_n],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Match: %{x:.1f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Candidate Match Scores",
                        xaxis_title="Match Percentage (%)",
                        yaxis_title="Candidate",
                        height=max(400, len(top_n) * 60),
                        yaxis={'categoryorder':'total ascending'},
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.markdown("---")
                    st.subheader("📥 Export Results")
                    
                    df_results = pd.DataFrame([{
                        'Rank': i+1,
                        'Name': c['Name'],
                        'Email': c['Email'],
                        'Phone': c['Phone'],
                        'Skills': ', '.join(c['Skills']) if c['Skills'] else 'None',
                        'Match Score (%)': f"{c['SimilarityScore']*100:.2f}",
                        'File': c['FileName']
                    } for i, c in enumerate(ranked_candidates[:num_candidates])])
                    
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="resume_screening_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit & Sentence-BERT | Resume Screening Tool v1.0</p>
        <p><small>Tip: For best results, provide detailed job descriptions with specific skills and requirements</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
