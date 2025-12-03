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
                name = m.name.replace("models/", "")
                models.append(name)
        return sorted(models, reverse=True) 
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
    
    # Handle missing script gracefully within the strict prompt structure
    script_text_to_inject = script_content if script_content and len(script_content.strip()) > 10 else "NOT PROVIDED (Perform Blind Audit based on visual/audio logic)"

    model = genai.GenerativeModel(
        model_name=model_id,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": list[AuditIssue],
            "temperature": 0.1, 
        }
    )

    # --- THE UPDATED FORENSIC PROMPT ---
    prompt = f"""
    Role: You are a Senior Video Quality Assurance (QA) Auditor for a premium broadcast production house. Your auditing style is forensic, strict, and detail-oriented. You do not skim; you analyze every frame, audio cue, and data point against the provided "Ground Truth" (Script & Fact Sheet).
    
    Objective: Conduct a minute-by-minute audit of the uploaded video file. Your goal is to catch every error—visual, audio, factual, or compliance-related—before the video goes to final export.
    
    Input Data Required:
    - The Video File: (Uploaded)
    - The Approved Script: {script_text_to_inject}
    - Key Facts/Data: General Indian Context (Use real-world Indian GST Laws, Tax Slabs, and ICMR Guidelines).
    
    The "Forensic" Checklist (Audit Parameters):
    
    1. STRICT SCRIPT FIDELITY & DEVIATIONS
    - Verbatim Check: Compare spoken dialogue against the script. Flag any skipped lines, ad-libs, or missing segments (e.g., skipped scenes).
    - Placeholder Detection: Flag if the script contains placeholder text (e.g., "lorem ipsum" or gibberish) but the video has actual dialogue.
    - Ending/Structure: Ensure the video ends exactly as scripted. If the script calls for a "Call to Action" (CTA) and the video ends with a skit/blooper instead, flag it as a Major Deviation.
    
    2. FACTUAL INTEGRITY & REAL-WORLD LOGIC
    - Data Validation (CRITICAL): Verify all numbers, percentages, and financial claims (e.g., GST rates, Tax slabs) against Real-World Laws (specifically Indian context).
    - Example: If the video says "4% GST" but the real tax slab is 5% or 18%, flag this as CRITICAL MISINFORMATION, even if the script is vague.
    - Grammage & Science: Check if visual data (e.g., "50g protein") aligns with the spoken narration or cited sources (e.g., ICMR guidelines).
    
    3. VISUAL FORENSICS & BRANDING
    - Text Legibility: strict check on all on-screen text. Is there sufficient contrast? (e.g., White text on a light background). Note: Be precise. Do not flag if the text is on a dark colored pill/shape.
    - Prop Logic & Continuity: Watch for objects appearing out of nowhere (e.g., an egg appearing in hand without an establishing "taking out of pocket" shot).
    - Branding & Parody: If the video uses parody brands (e.g., "Bomato" instead of "Zomato"), ensure the Trade Dress (colors, UI layout) isn't identical to the real brand to avoid legal risk.
    
    4. COMPLIANCE & PRIVACY (PII)
    - Screen Inspections (CRITICAL): Pause and zoom in on every shot featuring a phone or computer screen.
    - Check for visible PII (Personally Identifiable Information): Real names, phone numbers, email addresses, or delivery locations.
    - Action: If found, strict instruction to BLUR.
    
    5. AUDIO & PACING
    - Mix Levels: Ensure background music does not drown out the Voiceover (VO), especially during "reveal" or "cheering" moments.
    - Lip Sync: Verify audio matches speaker mouth movement.
    
    Output Format:
    Please present your findings in a structured JSON list format using these keys: 
    timestamp, severity (Critical/Major/Minor), category, issue_description, suggested_fix.
    """

    # Increased timeout to 10 minutes for long videos
    response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
    return response.text

# --- Sidebar: Model Selector ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    if "available_models" not in st.session_state:
        st.session_state.available_models = [
            "gemini-2.5-pro-preview-03-25", # High IQ
            "gemini-2.5-flash",             # High Speed
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-exp",
            "Custom Input"
        ]

    if st.button("🔄 Fetch All Available Models", help="Click to update the dropdown with all models your API key can access"):
        with st.spinner("Querying Google API for models..."):
            fetched = get_available_models()
            if fetched and not fetched[0].startswith("Error"):
                if "Custom Input" not in fetched:
                    fetched.append("Custom Input")
                st.session_state.available_models = fetched
                st.success(f"Found {len(fetched)-1} models!")
            else:
                st.error("Could not fetch models.")

    selection = st.selectbox("Select AI Model", st.session_state.available_models, index=0)
    
    if selection == "Custom Input":
        selected_model = st.text_input("Enter Model ID (e.g., gemini-3.0-pro)", "gemini-1.5-pro")
    else:
        selected_model = selection
        
    st.caption(f"Active Model ID: `{selected_model}`")

# --- Main UI ---
st.title("🕵️‍♂️ AI Video QA Auditor")
st.markdown("Upload your video to generate a forensic QA log. Script is optional but recommended.")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("1. Upload Assets")
    video_upload = st.file_uploader("Upload Video (MP4/MOV)", type=["mp4", "mov"])
    
    # Script is now purely optional UI
    with st.expander("📝 Optional: Add Script for comparison"):
        tab1, tab2 = st.tabs(["Paste Script", "Upload Script"])
        script_text = ""
        with tab1:
            script_text_paste = st.text_area("Paste Approved Script here", height=150)
        with tab2:
            script_file = st.file_uploader("Upload Script file", type=["txt", "md", "srt"])
        
        if script_file:
            script_text = script_file.read().decode("utf-8")
        elif script_text_paste:
            script_text = script_text_paste

    # Button enabled if Video is present (Script not required)
    start_audit = st.button("🚀 Run Forensic Audit", type="primary", use_container_width=True, disabled=not video_upload)

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
                with st.spinner(f"🧠 {selected_model} is analyzing content frame-by-frame..."):
                    try:
                        json_result = run_audit(gemini_video, script_text, selected_model)
                        data = json.loads(json_result)
                        st.session_state.audit_df = pd.DataFrame(data)
                        
                        # Cleanup
                        genai.delete_file(gemini_video.name)
                        st.success("Audit Complete!")
                        
                    except Exception as e:
                        st.error(f"Analysis Error: {e}")
                        
                        # Friendly Error Handling for 429
                        if "429" in str(e):
                             st.error("⚠️ Rate Limit Exceeded: The selected model is busy. Please switch to 'gemini-2.5-flash' in the sidebar for higher quota.")
                        elif "404" in str(e):
                             st.warning("⚠️ Model Not Found: Your API Key might not have access to this specific experimental model. Try 'gemini-1.5-flash'.")

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
        st.info("Waiting for video input...")
