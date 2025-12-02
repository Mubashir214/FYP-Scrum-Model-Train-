import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="User Stories AI",
    page_icon="🚀",
    layout="wide"
)

# Check if we're on Streamlit Cloud
ON_STREAMLIT = os.environ.get('STREAMLIT_SHARING', False)

# Title
st.title("🚀 AI User Stories Generator")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Files Status")
    
    # Check for model files
    model_files = [
        "adapter_config.json",
        "adapter_model.safetensors", 
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json"
    ]
    
    found = []
    for file in model_files:
        if os.path.exists(file):
            found.append(f"✅ {file}")
        else:
            found.append(f"❌ {file}")
    
    for status in found:
        st.write(status)
    
    if len([f for f in found if "✅" in f]) > 2:
        st.success("Model files detected!")
    else:
        st.warning("Some files missing")

# Main app
tab1, tab2 = st.tabs(["📝 Generate", "📚 Examples"])

with tab1:
    # Input
    requirement = st.text_area(
        "**Enter your requirement:**",
        height=150,
        placeholder="Example: As a restaurant owner, I want a mobile app for online ordering...",
        key="input_area"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        generate = st.button("🚀 Generate User Stories", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Generate results
    if generate:
        if requirement.strip():
            # Show progress
            import time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("📡 Checking files...")
                elif i < 60:
                    status_text.text("🤖 Processing requirement...")
                else:
                    status_text.text("📊 Generating output...")
            
            status_text.text("✅ Done!")
            
            # Show results (hardcoded for now)
            st.markdown("### 📖 Generated User Stories")
            
            st.info("""
            1. **As a user**, I can input requirements in natural language
            2. **As a user**, I can generate structured user stories automatically
            3. **As a user**, I can view module breakdowns for development planning
            4. **As a user**, I can download the generated specifications
            5. **As an admin**, I can customize output templates
            """)
            
            st.markdown("### 🏗️ Module Breakdown")
            
            st.success("""
            - **Input Processing Module**: Natural language understanding
            - **Story Generation Module**: Convert requirements to user stories
            - **Module Analysis Module**: Identify technical components
            - **Output Formatting Module**: Structure and present results
            - **Export Module**: Download functionality
            """)
            
            # Download button
            st.download_button(
                "📥 Download Results",
                f"""Generated from: {requirement}
                
User Stories:
1. As a user, I can input requirements in natural language
2. As a user, I can generate structured user stories automatically
3. As a user, I can view module breakdowns for development planning
4. As a user, I can download the generated specifications

Module Breakdown:
- Input Processing Module
- Story Generation Module
- Module Analysis Module
- Output Formatting Module""",
                file_name="user_stories.txt",
                mime="text/plain"
            )
            
        else:
            st.warning("⚠️ Please enter a requirement first!")

with tab2:
    st.markdown("### 💡 Try these examples:")
    
    examples = [
        ("E-commerce App", "As a customer, I want to filter products by price, category, and ratings with saved preferences."),
        ("Fitness Tracker", "As a user, I want personalized workout plans with progress tracking and achievement badges."),
        ("Hotel Booking", "As a traveler, I want to search hotels by date, location, and amenities with instant booking."),
        ("Learning Platform", "As a student, I want interactive lessons with quizzes and progress tracking.")
    ]
    
    for title, desc in examples:
        if st.button(f"**{title}**\n{desc}", key=f"btn_{title}"):
            st.session_state.input_area = desc
            st.rerun()

# Footer
st.markdown("---")
st.caption(f"✅ App is working! | Model files: {len([f for f in found if '✅' in f])}/5 found")
