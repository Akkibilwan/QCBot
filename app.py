import streamlit as st
import google.generativeai as genai
import time
import pandas as pd
import json
import os
from io import StringIO

# --- Page Config ---
st.set_page_config(page_title="Video QA Auditor AI", page_icon="🎬", layout="wide")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("---")
    st.info("💡 **Note:** Uploading large videos may take time depending on your internet speed.")
    st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")

# --- Helper Functions ---
def wait_for_files_active(files):
    """Waits for the uploaded video to be processed by Gemini."""
    print("Waiting for file processing...")
    bar = st.progress(0, text="Processing video on Gemini server...")
    for name in (file.name for file in files):
        file_obj = genai.get_file(name)
        while file_obj.state.name == "PROCESSING":
            time.sleep(2)  # Polling interval
            file_obj = genai.get_file(name)
        if file_obj.state.name != "ACTIVE":
            st.error(f"File {file_obj.name} failed to process.")
            return False
    bar.progress(100, text="Video processed and ready!")
    time.sleep(0.5)
    bar.empty()
    return True

def analyze_video(video_file, script_content, api_key):
    """Uploads video and script to Gemini and requests JSON analysis."""
    genai.configure(api_key=api_key)
    
    # 1. Upload Video to Gemini
    with st.spinner(f"Uploading {video_file.name} to Gemini..."):
        # Create a temporary file to handle the stream
        temp_filename = "temp_video.mp4"
        with open(temp_filename, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Upload using the path
        gemini_file = genai.upload_file(path=temp_filename, display_name=video_file.name)
        
        # Verify state
        if not wait_for_files_active([gemini_file]):
            return None

    # 2. Define the Prompt
    # We enforce specific JSON structure for dataframe compatibility
    system_prompt = """
    Role: You are a Senior Video Quality Assurance (QA) Auditor.
    Task: Audit the provided video against the provided script/ground truth.
    
    You must check for:
    1. Visual Integrity (Spelling, Text Safety, Glitches, B-roll relevance).
    2. Audio & Pacing (Lip Sync, Levels, Pronunciation, Dead Air).
    3. Factual Accuracy (Compare spoken audio verbatim to script, check data visuals).
    4. Compliance (PII, Watermarks).

    Output Format:
    Return ONLY a JSON array of objects. Each object must have these keys:
    - "timestamp": string (e.g., "01:23")
    - "severity": string ("Critical", "Major", "Minor", "Pass")
    - "category": string ("Visual", "Audio", "Fact", "Spelling", "Compliance")
    - "issue_description": string (Description of what is wrong, or "No errors found" if strictly requested)
    - "suggested_fix": string (Actionable advice)
    """

    user_prompt = f"""
    Here is the APPROVED SCRIPT and GROUND TRUTH data:
    {script_content}

    Please analyze the video provided strictly against this text.
    """

    # 3. Generate Content
    # We use gemini-1.5-pro for best video reasoning capabilities
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={"response_mime_type": "application/json"}
    )

    with st.spinner("AI is watching the video and auditing against the script..."):
        response = model.generate_content([gemini_file, system_prompt, user_prompt])
    
    # Clean up local file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return response.text

# --- Main Interface ---
st.title("🎬 AI Video Quality Assurance (QA) Tool")
st.markdown("""
Upload your video and your script/truth-doc. The AI will watch the video, read the script, 
and generate a **detailed QA report** highlighting errors in spelling, facts, audio, and visuals.
""")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_video = st.file_uploader("1️⃣ Upload Video (MP4, MOV)", type=["mp4", "mov", "avi"])

with col2:
    uploaded_script = st.file_uploader("2️⃣ Upload Script/Doc (TXT, MD)", type=["txt", "md"])

# Initialize session state for results
if "qa_results" not in st.session_state:
    st.session_state.qa_results = None

# --- Execution ---
if st.button("🚀 Run QA Analysis", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not uploaded_video or not uploaded_script:
        st.error("Please upload both a video file and a script file.")
    else:
        # Decode script
        try:
            stringio = StringIO(uploaded_script.getvalue().decode("utf-8"))
            script_text = stringio.read()
            
            # Run Analysis
            json_response = analyze_video(uploaded_video, script_text, api_key)
            
            if json_response:
                try:
                    data = json.loads(json_response)
                    st.session_state.qa_results = data
                except json.JSONDecodeError:
                    st.error("Failed to parse AI response. Raw output provided below.")
                    st.text(json_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Results Display ---
if st.session_state.qa_results:
    st.divider()
    st.subheader("📋 QA Audit Report")
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.qa_results)
    
    # Styling the dataframe for better readability
    def highlight_severity(val):
        color = 'white'
        if val == 'Critical': color = '#ff4b4b' # Red
        elif val == 'Major': color = '#ffa421'  # Orange
        elif val == 'Minor': color = '#ffe169'  # Yellow
        elif val == 'Pass': color = '#21c354'   # Green
        return f'background-color: {color}; color: black'

    if not df.empty:
        # Reorder columns if needed
        cols = ["timestamp", "severity", "category", "issue_description", "suggested_fix"]
        # Ensure columns exist before selecting
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]

        st.dataframe(
            df.style.applymap(highlight_severity, subset=['severity']),
            use_container_width=True,
            hide_index=True
        )
        
        # CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv,
            file_name="qa_audit_report.csv",
            mime="text/csv",
        )
    else:
        st.success("Analysis complete. No structural errors found based on JSON output.")
