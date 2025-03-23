# Required Libraries
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import PyPDF2

# Load Pre-trained BERT Model for Semantic Similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample Job Description
job_description = """
We are looking for a Python Developer with experience in machine learning, natural language processing (NLP),
and data analysis. Proficiency in Python, TensorFlow, and NLP libraries like NLTK or spaCy is required.
"""

# Function to Extract Text from PDF Resume
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Folder containing resumes (PDFs)
resume_folder = "resumes/"

# Store Results
candidates = []

# Iterate through resumes
for resume_file in os.listdir(resume_folder):
    if resume_file.endswith(".pdf"):
        resume_path = os.path.join(resume_folder, resume_file)
        resume_text = extract_text_from_pdf(resume_path)

        # Encode job description and resume using BERT
        jd_embedding = model.encode(job_description, convert_to_tensor=True)
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)

        # Compute Semantic Similarity
        similarity_score = util.cos_sim(jd_embedding, resume_embedding).item()

        candidates.append({
            "Candidate": resume_file.replace('.pdf', ''),
            "Similarity Score": round(similarity_score * 100, 2)
        })

# Convert to DataFrame and Rank Candidates
df = pd.DataFrame(candidates)
df.sort_values(by="Similarity Score", ascending=False, inplace=True)

# Output Ranked Candidates
print("\nRanked Candidates based on Job Description Matching:\n")
print(df.reset_index(drop=True))
