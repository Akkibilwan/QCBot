import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import os
import tempfile
import json
import typing_extensions as typing

# --- Configuration ---
st.set_page_config(page_title="Forensic Video QA Auditor", page_icon="🕵️‍♂️", layout="wide")

# --- CSS for styling ---
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Secrets Management ---
# Safely access API key from .streamlit/secrets.toml
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Please set your GEMINI_API_KEY in .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)

# --- Define Output Schema (Structured Output) ---
# This ensures the API returns JSON exactly matching this structure
class AuditIssue(typing.TypedDict):
    timestamp: str
    severity: str
    category: str
    issue_description: str
    suggested_fix: str

# --- Helper Functions ---

def upload_to_gemini(uploaded_file):
    """Writes bytes to temp file and uploads to Gemini."""
    with st.spinner(f"Processing video: {uploaded_file.name}..."):
        # Create a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) 
        tfile.write(uploaded_file.read())
        tfile.close()

        # Upload to Gemini
        gemini_file = genai.upload_file(tfile.name, mime_type="video/mp4")
        
        # Poll for processing completion
        while gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = genai.get_file(gemini_file.name)
        
        if gemini_file.state.name == "FAILED":
            st.error("Video processing failed on Google's server.")
            return None

        # Clean up local temp file
        os.unlink(tfile.name)
        return gemini_file

def run_audit(video_file, script_content):
    """Runs the QA prompts against the video and script."""
    
    # We use Flash for speed/cost, or Pro for deeper reasoning. 
    # Flash 1.5 is excellent for video analysis.
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": list[AuditIssue] # Enforce list of objects
        }
    )

    # The Master Prompt
    prompt = f"""
    You are a Senior Video Quality Assurance (QA) Auditor. Your auditing style is forensic.
    
    OBJECTIVE:
    Conduct a minute-by-minute audit of the video against the provided script. catch every visual, audio, factual, or compliance error.

    INPUT CONTEXT:
    1. **Approved Script:** {script_content}

    AUDIT PARAMETERS (The Forensic Checklist):
    1. **Strict Script Fidelity:** Flag skipped lines, ad-libs, or placeholder text (gibberish).
    2. **Factual Integrity:** CRITICAL. Verify all numbers (GST rates, Tax slabs, Stats). If video says "4% GST" but reality is 5% or 18%, flag as CRITICAL MISINFORMATION.
    3. **Visual Forensics:** Check text legibility (contrast), prop continuity, and PII leaks on phone screens (Names/Numbers).
    4. **Compliance:** Ensure parody brands (e.g., "Bomato") don't violate trade dress.
    5. **Audio:** Check mix levels and lip sync.

    OUTPUT INSTRUCTION:
    Analyze the entire video. Return a JSON list of issues found.
    """

    with st.spinner("🤖 AI Auditor is analyzing every frame... this may take a moment."):
        response = model.generate_content([video_file, prompt])
        return response.text

# --- Main UI ---

st.title("🕵️‍♂️ AI Video QA Auditor")
st.markdown("Upload your video and the approved script to generate a forensic QA log.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Assets")
    video_upload = st.file_uploader("Upload Video (MP4)", type=["mp4", "mov"])
    
    script_mode = st.radio("Script Input Method:", ["Paste Text", "Upload Text/SRT File"])
    
    script_text = ""
    if script_mode == "Paste Text":
        script_text = st.text_area("Paste Approved Script/Context here", height=200)
    else:
        script_file = st.file_uploader("Upload Script", type=["txt", "md", "srt"])
        if script_file:
            script_text = script_file.read().decode("utf-8")

    analyze_btn = st.button("Run Forensic Audit", type="primary", disabled=not (video_upload and script_text))

with col2:
    st.subheader("2. Audit Results")
    
    if "audit_data" not in st.session_state:
        st.session_state.audit_data = None

    if analyze_btn:
        # 1. Upload Video
        gemini_video = upload_to_gemini(video_upload)
        
        if gemini_video:
            # 2. Analyze
            try:
                json_result = run_audit(gemini_video, script_text)
                data = json.loads(json_result)
                st.session_state.audit_data = pd.DataFrame(data)
                
                # Cleanup Gemini file to save storage limit
                genai.delete_file(gemini_video.name)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    # Display Results
    if st.session_state.audit_data is not None:
        df = st.session_state.audit_data
        
        # Severity Color Highlight
        def highlight_severity(val):
            color = ''
            if 'Critical' in val or 'CRITICAL' in val:
                color = 'background-color: #ffcccc; color: #990000; font-weight: bold;'
            elif 'Major' in val or 'MAJOR' in val:
                color = 'background-color: #fff4cc; color: #664400;'
            return color

        st.dataframe(
            df.style.map(highlight_severity, subset=['severity']),
            use_container_width=True,
            column_config={
                "timestamp": "Time",
                "severity": "Severity",
                "category": "Category",
                "issue_description": "Issue",
                "suggested_fix": "Suggested Fix"
            }
        )

        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download QA Report (CSV)",
            csv,
            "QA_Audit_Log.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("Upload files and click 'Run Forensic Audit' to see results.")
