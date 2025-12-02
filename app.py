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
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Secrets Management ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("Please set your GEMINI_API_KEY in .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)

# --- Define Output Schema (Structured JSON) ---
class AuditIssue(typing.TypedDict):
    timestamp: str
    severity: str
    category: str
    issue_description: str
    suggested_fix: str

# --- Helper Functions ---

def get_available_models():
    """Fetches list of models available to your API key."""
    try:
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(m.name)
        return sorted(models)
    except Exception as e:
        return [f"Error fetching models: {e}"]

def upload_to_gemini(uploaded_file):
    """Uploads video to Gemini and polls until active."""
    with st.spinner(f"📤 Uploading & Processing Video: {uploaded_file.name}..."):
        # Create temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) 
        tfile.write(uploaded_file.read())
        tfile.close()

        try:
            # Upload
            gemini_file = genai.upload_file(tfile.name, mime_type="video/mp4")
            
            # Poll state
            bar = st.progress(0, text="Processing video on Google Servers...")
            while gemini_file.state.name == "PROCESSING":
                time.sleep(2)
                gemini_file = genai.get_file(gemini_file.name)
                bar.progress(50)
            
            bar.empty()
            
            if gemini_file.state.name == "FAILED":
                st.error("❌ Video processing failed on Google's server.")
                return None
            
            return gemini_file
            
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return None
        finally:
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)

def run_audit(video_file, script_content, model_id):
    """Runs the forensic audit using the selected model."""
    
    model = genai.GenerativeModel(
        model_name=model_id,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": list[AuditIssue],
            "temperature": 0.1, # Low temp for strict factual adherence
        }
    )

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

    # Increased timeout to 10 minutes for long videos
    response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
    return response.text

# --- Sidebar: Model Selector ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # Pre-defined list of high-performance models
    model_options = [
        "gemini-1.5-pro-002",    # Latest Stable Pro
        "gemini-1.5-flash-002",  # Latest Stable Flash
        "gemini-1.5-pro",        # Legacy Pro
        "gemini-1.5-flash",      # Legacy Flash
        "gemini-exp-1121",       # Experimental (Check availability)
        "Custom Input"           # Future proofing
    ]
    
    selection = st.selectbox("Select AI Model", model_options, index=0)
    
    if selection == "Custom Input":
        selected_model = st.text_input("Enter Model ID (e.g., gemini-3.0-pro)", "gemini-1.5-pro-002")
    else:
        selected_model = selection
        
    st.info(f"Active Model: `{selected_model}`")
    
    with st.expander("🛠️ Debug: List My Models"):
        if st.button("Fetch Available Models"):
            with st.spinner("Fetching..."):
                available = get_available_models()
                st.code("\n".join(available), language="text")

# --- Main UI ---
st.title("🕵️‍♂️ AI Video QA Auditor")
st.markdown("Upload your video and script to generate a forensic QA log.")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1. Upload Assets")
    video_upload = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov"])
    
    tab1, tab2 = st.tabs(["Paste Script", "Upload Script"])
    script_text = ""
    
    with tab1:
        script_text_paste = st.text_area("Paste Approved Script here", height=250)
    with tab2:
        script_file = st.file_uploader("Upload Script file", type=["txt", "md", "srt"])
    
    # Logic to prioritize file over paste if both exist
    if script_file:
        script_text = script_file.read().decode("utf-8")
    elif script_text_paste:
        script_text = script_text_paste

    start_audit = st.button("🚀 Run Forensic Audit", type="primary", use_container_width=True, disabled=not (video_upload and script_text))

with col2:
    st.subheader("2. Audit Results")
    
    if "audit_df" not in st.session_state:
        st.session_state.audit_df = None

    if start_audit:
        if not API_KEY:
            st.error("API Key missing.")
        else:
            # 1. Upload
            gemini_video = upload_to_gemini(video_upload)
            
            if gemini_video:
                # 2. Audit
                with st.spinner(f"🧠 {selected_model} is analyzing content..."):
                    try:
                        json_result = run_audit(gemini_video, script_text, selected_model)
                        data = json.loads(json_result)
                        st.session_state.audit_df = pd.DataFrame(data)
                        
                        # Cleanup
                        genai.delete_file(gemini_video.name)
                        st.success("Audit Complete!")
                        
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
                        st.warning("If you see a 404, the selected model might not be available to your account yet. Try 'gemini-1.5-flash'.")

    # Display Data
    if st.session_state.audit_df is not None and not st.session_state.audit_df.empty:
        df = st.session_state.audit_df
        
        # Styling
        def color_severity(val):
            color = ''
            v = str(val).lower()
            if 'critical' in v:
                return 'background-color: #ffb3b3; color: black; font-weight: bold;'
            elif 'major' in v:
                return 'background-color: #fff4cc; color: black;'
            elif 'minor' in v:
                return 'color: gray;'
            return ''

        st.dataframe(
            df.style.map(color_severity, subset=['severity']),
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Time"),
                "severity": st.column_config.TextColumn("Severity"),
                "category": "Category",
                "issue_description": "Issue",
                "suggested_fix": "Fix"
            },
            hide_index=True
        )
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report (CSV)", csv, "QA_Audit.csv", "text/csv", use_container_width=True)

    elif st.session_state.audit_df is not None:
        st.success("✅ Clean Audit. No issues found.")
    else:
        st.info("Waiting for input...")
