import streamlit as st
import pickle
import spacy
import re
from wordcloud import WordCloud
from docx import Document
import fitz
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load ML model pipeline
with open('model_res.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Job category mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
}

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    clean = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(clean)

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# DOCX text extraction
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join(para.text for para in doc.paragraphs)

# Category prediction
def predict_category(text):
    cleaned_text = preprocess_text(text)
    prediction_id = model_pipeline.predict([cleaned_text])[0]
    return category_mapping.get(prediction_id, "Unknown")

# Contact info extraction
def extract_contact_info(text):
    email = re.findall(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", text)
    phone = re.findall(r"\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}", text)
    return email[0] if email else "Not Found", phone[0] if phone else "Not Found"

# Skill extraction
def extract_skills(text, skill_db=None):
    if not skill_db:
        skill_db =  {
    "python", "java", "sql", "excel", "machine learning", "deep learning", "data analysis",
    "django", "flask", "html", "css", "javascript", "project management", "communication",
    "leadership", "teamwork", "agile", "cloud computing", "aws", "azure", "docker", "kubernetes",
    "scrum", "data visualization", "bi", "spark", "hadoop", "tensorflow", "pytorch", "r", "sas",
    "postgresql", "react", "nodejs", "graphql", "swift", "android", "iOS", "salesforce", "SAP",
    "tableau", "power bi", "photoshop", "graphic design", "ruby", "php"
}
    words = set(word.lower() for word in preprocess_text(text).split())
    return list(words & skill_db)

# Download cleaned text
def get_download_link(text, filename="cleaned_resume.txt"):
    b = BytesIO()
    b.write(text.encode())
    b.seek(0)
    b64 = base64.b64encode(b.read()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Cleaned Resume Text</a>'

# Resume improvement suggestions for all job categories
def improvement_suggestions(resume_text, skills_found, email, phone, category):
    suggestions = []
    word_count = len(resume_text.split())

    # Check if the resume is already perfect, no suggestions if everything is ideal
    if word_count >= 300 and email != "Not Found" and re.match(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", email) and phone != "Not Found" and re.match(r"\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}", phone) and len(skills_found) >= 5:
        return suggestions  # Return empty list if no improvements are needed

    # General suggestions based on word count
    if word_count < 100:
        suggestions.append("➤ Consider adding more detail to your resume, especially in terms of experience and skills.")

    # Contact info suggestions
    if email == "Not Found":
        suggestions.append("➤ Include a valid email address for easier contact.")
    elif not re.match(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b", email):
        suggestions.append("➤ Please use a valid email format (e.g., example@domain.com).")

    if phone == "Not Found":
        suggestions.append("➤ Include a contact number.")
    elif not re.match(r"\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}", phone):
        suggestions.append("➤ Make sure your phone number is in a valid format.")

    # Skills-based suggestions
    if len(skills_found) < 3:
        suggestions.append("➤ Highlight more relevant skills (technical or soft). Consider adding specific skills related to your job role.")

    # Category-specific suggestions for various roles
    role_specific_suggestions = {
        "Python Developer": [
            "➤ Ensure you highlight experience with Python frameworks like Django or Flask.",
            "➤ Mention Machine Learning, Data Science, or automation tools (e.g., Pandas, NumPy, TensorFlow).",
            "➤ Include cloud services experience such as AWS, Azure, or Google Cloud."
        ],
        "Java Developer": [
            "➤ Add experience with Java frameworks like Spring or Hibernate.",
            "➤ Highlight experience in multi-threading and concurrency for performance optimization.",
            "➤ Mention version control systems like Git, and continuous integration tools."
        ],
        "Testing": [
            "➤ Specify your knowledge of testing frameworks (e.g., Selenium, JUnit, or TestNG).",
            "➤ Mention your experience with automated testing and CI/CD pipelines.",
            "➤ Include familiarity with test management tools like Jira or TestRail."
        ],
        "Data Science": [
            "➤ Mention specific data manipulation libraries like Pandas and NumPy.",
            "➤ Highlight experience with machine learning algorithms and libraries (e.g., Scikit-Learn, TensorFlow, or PyTorch).",
            "➤ Ensure to include knowledge of data visualization tools like Matplotlib, Seaborn, or Tableau."
        ],
        "DevOps Engineer": [
            "➤ Add experience with cloud platforms like AWS, Azure, or GCP.",
            "➤ Mention knowledge of CI/CD tools like Jenkins, GitLab CI, or CircleCI.",
            "➤ Specify experience with containerization tools such as Docker and orchestration tools like Kubernetes."
        ],
        "Web Designing": [
            "➤ Highlight experience with HTML, CSS, JavaScript, and front-end frameworks like React or Angular.",
            "➤ Mention your understanding of responsive design and web accessibility standards.",
            "➤ If applicable, include experience with back-end technologies like Node.js or PHP."
        ],
        "HR": [
            "➤ Highlight your experience with HRIS (Human Resource Information Systems) and ATS (Applicant Tracking Systems).",
            "➤ Mention knowledge of employee relations, recruitment, and performance management.",
            "➤ Add relevant certifications like SHRM or HRCI."
        ],
        "Blockchain": [
            "➤ Ensure to mention your knowledge of blockchain platforms like Ethereum, Hyperledger, or Solana.",
            "➤ Highlight experience with smart contracts, decentralized applications (dApps), and cryptocurrency.",
            "➤ Mention your understanding of consensus algorithms and cryptography."
        ],
        "Operations Manager": [
            "➤ Highlight leadership skills and experience with project management tools like Jira or Trello.",
            "➤ Add experience in process optimization, budgeting, and strategic planning.",
            "➤ Mention knowledge of supply chain management and resource planning."
        ],
        "Mechanical Engineer": [
            "➤ Ensure to include your expertise in CAD software (e.g., AutoCAD, SolidWorks, or CATIA).",
            "➤ Mention any experience with product lifecycle management (PLM) and manufacturing processes.",
            "➤ Highlight any certifications such as Six Sigma or Lean Manufacturing."
        ],
        "Sales": [
            "➤ Include experience with CRM tools like Salesforce or HubSpot.",
            "➤ Highlight your understanding of sales methodologies like SPIN Selling or BANT.",
            "➤ Add achievements in sales targets or quotas if applicable."
        ],
        "Database": [
            "➤ Mention experience with database management systems (e.g., MySQL, PostgreSQL, or MongoDB).",
            "➤ Highlight knowledge of database optimization, indexing, and query tuning.",
            "➤ Include familiarity with cloud-based databases and data warehousing solutions."
        ],
        "Network Security Engineer": [
            "➤ Add certifications like CISSP, CEH, or CompTIA Security+. ",
            "➤ Highlight your experience with firewalls, intrusion detection/prevention systems, and VPNs.",
            "➤ Mention familiarity with network protocols, encryption techniques, and security audits."
        ],
        "PMO": [
            "➤ Mention your experience with project management frameworks like PMI or PRINCE2.",
            "➤ Include experience with project scheduling tools like MS Project or Primavera.",
            "➤ Highlight your knowledge of risk management and change management."
        ],
        "Business Analyst": [
            "➤ Highlight experience in gathering business requirements, process mapping, and data analysis.",
            "➤ Mention your familiarity with tools like MS Visio, Tableau, or Power BI.",
            "➤ Include knowledge of Agile and Scrum methodologies."
        ],
        "Civil Engineer": [
            "➤ Include your experience with construction software like AutoCAD or Revit.",
            "➤ Mention your understanding of structural design, materials science, and project management.",
            "➤ Add certifications such as PE (Professional Engineer) or PMP (Project Management Professional)."
        ],
        "Advocate": [
            "➤ Mention specific legal areas of expertise, such as corporate law, family law, or criminal defense.",
            "➤ Highlight your familiarity with legal research tools and case management software.",
            "➤ Include bar association memberships and relevant certifications."
        ]
    }

    # Append role-specific suggestions
    if category in role_specific_suggestions:
        suggestions.extend(role_specific_suggestions[category])

    return suggestions


# Job Role Recommendation (Across different categories)
def recommend_jobs(resume_text):
    # Use category prediction for job roles
    cleaned_text = preprocess_text(resume_text)
    prediction_id = model_pipeline.predict([cleaned_text])[0]
    category = category_mapping.get(prediction_id, "Unknown")

    # List of different categories to recommend
    category_recommendations = {
        "Java Developer": ["HR", "Sales", "Operations Manager", "Mechanical Engineer", "PMO"],
        "Testing": ["Data Science", "Blockchain", "Business Analyst", "Civil Engineer", "Sales"],
        "DevOps Engineer": ["HR", "Project Management", "Data Scientist", "Operations Manager", "Mechanical Engineer"],
        "Python Developer": ["HR", "Marketing", "Database", "Sales", "Network Security Engineer"],
        "Web Designing": ["Data Science", "HR", "Sales", "Civil Engineer", "Mechanical Engineer"],
        "HR": ["Data Science", "Operations Manager", "Sales", "Backend Developer", "Mechanical Engineer"],
        "Data Science": ["Sales", "Web Designing", "Operations Manager", "Mechanical Engineer", "Blockchain"],
        "Database": ["HR", "Operations Manager", "Sales", "Marketing", "Civil Engineer"],
        "Sales": ["Data Science", "Web Designing", "HR", "Backend Developer", "Mechanical Engineer"],
        "Mechanical Engineer": ["Data Science", "HR", "Sales", "PMO", "Database"],
    }

    # Fetch related roles for the predicted category
    recommendations = category_recommendations.get(category, [category])

    return recommendations[:3]  # Return top 3 different job categories

# Streamlit app
def main():
    st.title("Resume Screening and Improvement App")

    uploaded_file = st.file_uploader("Upload Your Resume (PDF, DOCX, or Text file)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        analyze_button = st.button("🔍 Analyze Resume")

        if analyze_button:
            with st.spinner("Analyzing..."):
                # Extract text based on the file type
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    resume_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return

                # Predict job category
                category = predict_category(resume_text)
                email, phone = extract_contact_info(resume_text)
                skills = extract_skills(resume_text)
                cleaned_text = preprocess_text(resume_text)

                st.success(f"✅ Predicted Job Role: **{category}**")
                st.info(f"📧 Email: {email}  \n📱 Phone: {phone}")
                st.markdown(f"**💡 Top Skills Detected:** {', '.join(skills) if skills else 'None'}")

                # Get improvement suggestions
                suggestions = improvement_suggestions(resume_text, skills, email, phone, category)
                if suggestions:
                    st.subheader("🛠️ Suggestions to Improve Your Resume")
                    for s in suggestions:
                        st.warning(s)

                # Word cloud visualization
                st.subheader("🔤 Word Cloud")
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

                # Provide a download link for the cleaned resume text
                st.markdown(get_download_link(cleaned_text), unsafe_allow_html=True)

                # Recommend top job roles across different categories
                recommendations = recommend_jobs(resume_text)
                st.subheader("💼 Top Job Recommendations (Different Categories)")
                st.write(", ".join(recommendations))

if __name__ == "__main__":
    main()
