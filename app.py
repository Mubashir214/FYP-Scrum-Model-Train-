import streamlit as st
import time
import os

# ========== SIMPLE APP ==========
st.set_page_config(page_title="User Stories Generator", layout="wide")

st.title("🚀 AI User Stories Generator")
st.markdown("---")

# Quick check for model files
st.sidebar.header("Model Status")
if os.path.exists("finetuned_lora"):
    st.sidebar.success("✅ Model files found")
else:
    st.sidebar.error("❌ Model files missing")

# Simple input
requirement = st.text_area(
    "Enter your requirement:",
    height=150,
    placeholder="Example: As a project manager, I need a dashboard to track team progress..."
)

# Generate button
if st.button("Generate User Stories", type="primary"):
    if requirement.strip():
        with st.spinner("Generating..."):
            time.sleep(2)  # Simulate processing
            
            # Show placeholder results (for testing)
            st.markdown("### 📖 Generated User Stories:")
            st.markdown("""
            1. **As a project manager**, I can view real-time team progress on a centralized dashboard
            2. **As a project manager**, I can set and track project deadlines with visual indicators
            3. **As a team member**, I can update my task status with progress percentage
            4. **As a stakeholder**, I can receive automated weekly progress reports
            """)
            
            st.markdown("### 🏗️ Module Breakdown:")
            st.markdown("""
            - **Dashboard Module**: Real-time progress visualization
            - **Task Management Module**: Create, assign, and track tasks
            - **Reporting Module**: Automated report generation
            - **Notification Module**: Deadline reminders and alerts
            """)
            
            st.success("✅ Generation complete!")
    else:
        st.warning("Please enter a requirement")

# Footer
st.markdown("---")
st.caption("This is a simplified version for deployment testing")
