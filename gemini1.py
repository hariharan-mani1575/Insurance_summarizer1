import streamlit as st
import json
import requests
import io
import PyPDF2 # Import PyPDF2 for PDF handling
import os # Import os for accessing environment variables (used by st.secrets internally)

# --- Configuration ---
# Access the API key securely from Streamlit's secrets
try:
    # Try to get API key from st.secrets (recommended for Streamlit Cloud)
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback to environment variable for local development if not using secrets.toml
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please set it in Streamlit secrets or as an environment variable.")
        st.stop() # Stop the app if API key is missing

# Function to call the Gemini API for summarization
def summarize_document(document_text: str):
    """
    Summarizes the provided document text using the Gemini 2.0 Flash model,
    extracting coverages, exclusions, and policy details in a structured JSON format.
    """
    st.info("Analyzing document and generating summary...")

    chat_history = []
    prompt = f"""
    You are an AI assistant specialized in summarizing insurance documents.
    Please read the following insurance document and extract the following information in a structured JSON format:
    1.  A concise overall 'summary' of the document.
    2.  A list of 'coverages' provided by the policy.
    3.  A list of 'exclusions' (what is not covered) by the policy.
    4.  'policyDetails' as an object containing:
        -   'policyNumber' (if found)
        -   'policyHolder' (if found)
        -   'effectiveDate' (if found, e.g., "YYYY-MM-DD")
        -   'expirationDate' (if found, e.g., "YYYY-MM-DD")
        -   'premium' (if found, e.g., "USD 1200" or "1200 per year")
        -   'otherDetails' (a list of any other significant policy details not covered above).

    If a piece of information is not explicitly found, use "N/A" for strings or an empty list for arrays.
    Ensure the output is valid JSON.

    Document:
    ---
    {document_text}
    ---
    """
    chat_history.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "STRING"},
                    "coverages": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "exclusions": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "policyDetails": {
                        "type": "OBJECT",
                        "properties": {
                            "policyNumber": {"type": "STRING"},
                            "policyHolder": {"type": "STRING"},
                            "effectiveDate": {"type": "STRING"},
                            "expirationDate": {"type": "STRING"},
                            "premium": {"type": "STRING"},
                            "otherDetails": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            }
                        },
                        "propertyOrdering": ["policyNumber", "policyHolder", "effectiveDate", "expirationDate", "premium", "otherDetails"]
                    }
                },
                "propertyOrdering": ["summary", "coverages", "exclusions", "policyDetails"]
            }
        }
    }

    # Use the securely fetched API key
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            json_string = result["candidates"][0]["content"]["parts"][0]["text"]
            # The API might return the JSON string with markdown backticks, remove them if present
            if json_string.startswith("```json") and json_string.endswith("```"):
                json_string = json_string[7:-3].strip()
            return json.loads(json_string)
        else:
            st.error("Error: Could not get a valid response from the summarization model.")
            st.json(result) # Show the raw result for debugging
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network or API error: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON response from model: {e}")
        st.text(response.text) # Show the raw text response for debugging
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to extract text from PDF files
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Insurance Document Summarizer", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #1e3a8a; /* Dark blue */
        text-align: center;
        margin-bottom: 30px;
    }
    .stFileUploader > div > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stFileUploader > div > button:hover {
        background-color: #45a049;
    }
    .stButton > button {
        background-color: #1e3a8a; /* Dark blue */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        display: block;
        margin: 20px auto;
    }
    .stButton > button:hover {
        background-color: #15306b;
    }
    .summary-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
    }
    .summary-section h3 {
        color: #1e3a8a;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .summary-section ul {
        list-style-type: disc;
        margin-left: 20px;
        padding-left: 0;
    }
    .summary-section li {
        margin-bottom: 8px;
        color: #333;
    }
    .policy-details-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
    }
    .policy-details-table th, .policy-details-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .policy-details-table th {
        background-color: #f2f2f2;
        color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 Insurance Document Summarizer")
st.write("Upload your insurance document (text file or PDF) and get a quick summary of its key details.")

uploaded_file = st.file_uploader("Choose a document (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    file_content = None
    if uploaded_file.type == "text/plain":
        file_content = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.type == "application/pdf":
        file_content = extract_text_from_pdf(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")

    if file_content:
        st.subheader("Uploaded Document Preview:")
        with st.expander("Click to view document content"):
            st.text_area("Document Content", file_content, height=300, disabled=True)

        if st.button("Summarize Document"):
            if file_content:
                with st.spinner("Summarizing your document... This may take a moment."):
                    # Clear previous summary if a new file is uploaded or button is clicked again
                    if 'summary_data' in st.session_state:
                        del st.session_state['summary_data']

                    summary_data = summarize_document(file_content)
                    st.session_state['summary_data'] = summary_data # Store in session state

                    if summary_data:
                        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
                        st.subheader("Summary Report")

                        st.markdown("### Overall Summary")
                        st.write(summary_data.get("summary", "N/A"))

                        st.markdown("### Coverages")
                        if summary_data.get("coverages"):
                            for coverage in summary_data["coverages"]:
                                st.markdown(f"- {coverage}")
                        else:
                            st.write("No specific coverages found or identified.")

                        st.markdown("### Exclusions")
                        if summary_data.get("exclusions"):
                            for exclusion in summary_data["exclusions"]:
                                st.markdown(f"- {exclusion}")
                        else:
                            st.write("No specific exclusions found or identified.")

                        st.markdown("### Policy Details")
                        policy_details = summary_data.get("policyDetails", {})
                        if policy_details:
                            st.markdown(f"""
                            <table class="policy-details-table">
                                <tr><th>Detail</th><th>Value</th></tr>
                                <tr><td>Policy Number</td><td>{policy_details.get("policyNumber", "N/A")}</td></tr>
                                <tr><td>Policy Holder</td><td>{policy_details.get("policyHolder", "N/A")}</td></tr>
                                <tr><td>Effective Date</td><td>{policy_details.get("effectiveDate", "N/A")}</td></tr>
                                <tr><td>Expiration Date</td><td>{policy_details.get("expirationDate", "N/A")}</td></tr>
                                <tr><td>Premium</td><td>{policy_details.get("premium", "N/A")}</td></tr>
                            </table>
                            """, unsafe_allow_html=True)

                            if policy_details.get("otherDetails"):
                                st.markdown("#### Other Policy Details")
                                for detail in policy_details["otherDetails"]:
                                    st.markdown(f"- {detail}")
                        else:
                            st.write("No specific policy details found or identified.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate summary. Please try again or check the document content.")
            else:
                st.warning("Could not extract content from the uploaded document.")
    else:
        st.warning("Please upload a document to summarize.")
else:
    st.info("Upload a .txt or .pdf file to begin summarization.")
    st.markdown("""
    **Note:** This application now supports plain text files (.txt) and PDF files (.pdf).
    For best results, ensure your document is clear and well-formatted.
    """)