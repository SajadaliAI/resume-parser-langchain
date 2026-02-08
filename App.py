
import os
import json
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.prompts import PromptTemplate  

# ------------------------
# Load API Key and Configure LLM
# ------------------------
load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ------------------------
# Prompt Template
# ------------------------
PROMPT_TEMPLATE = """
You are an expert resume parser. Given the resume text, extract the following fields and return a single valid JSON object:

{{
  "Name": "...",
  "Email": "...",
  "Phone": "...",
  "LinkedIn": "...",
  "Skills": [...],
  "Education": [...],
  "Experience": [...],
  "Projects": [...],
  "Certifications": [...],
  "Languages": [...]
}}

Rules:
- If a field cannot be found, set its value to "No idea".
- Return ONLY valid JSON (no extra commentary).
- Keep lists as arrays, and keep Experience/Projects as arrays of short strings.

Resume text:
{text}
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])

# ------------------------
# Helper function to load files
# ------------------------
def extract_file(uploaded_file):
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    # Load file according to type
    name_lower = uploaded_file.name.lower()
    if name_lower.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif name_lower.endswith(".txt"):
        loader = TextLoader(temp_path)
    elif name_lower.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    else:
        return None

    return loader.load()

# ------------------------
# Streamlit App
# ------------------------
def main():
    st.set_page_config(page_title="Resume Parser", layout="wide")
    st.title("📄 Resume Parser using LangChain & Groq")

    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])

    if uploaded_file:
        with st.spinner("Uploading and extracting text..."):
            docs = extract_file(uploaded_file)
            if not docs:
                st.error("Unsupported file type!")
                st.stop()

        st.subheader("Extracted Text (Preview)")
        preview_text = "\n\n".join([d.page_content for d in docs])[:4000]
        st.text_area("Preview", value=preview_text, height=200)

        if st.button("Parse Resume"):
            with st.spinner("Sending to LLM..."):
                full_text = "\n\n".join([d.page_content for d in docs])
                formatted_prompt = prompt.format(text=full_text)

                try:
                    response = llm.invoke(formatted_prompt)
                    response_text = response.content if hasattr(response, "content") else str(response)
                    parsed_json = json.loads(response_text)
                    st.subheader("Parsed Resume (JSON)")
                    st.json(parsed_json)
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON from LLM response:")
                    st.write(response_text)

# ------------------------
if __name__ == "__main__":
    main()
