import streamlit as st
import torch
import time
import os

# Set page config
st.set_page_config(
    page_title="AI User Stories Generator",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 AI User Stories Generator")
st.markdown("Transform requirements into user stories and module breakdown")

# Check if files exist
st.sidebar.header("📁 File Status")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json"
]

for file in required_files:
    if os.path.exists(file):
        st.sidebar.success(f"✅ {file}")
    else:
        st.sidebar.error(f"❌ {file}")

# Simple input
requirement = st.text_area(
    "**Enter your user requirement:**",
    height=150,
    placeholder="Example: As a restaurant owner, I want a mobile app for customers to view menu and place orders online..."
)

# Settings in sidebar
st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
max_length = st.sidebar.slider("Max Length", 200, 800, 500, 50)

# Generate button
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    generate = st.button("🚀 Generate User Stories", type="primary", use_container_width=True)

# Examples
with st.expander("💡 Example Requirements", expanded=True):
    examples = [
        "As a project manager, I need a dashboard to track team progress, deadlines, and resource allocation.",
        "As a fitness coach, I want to create personalized workout plans and track client progress.",
        "As an e-commerce business, we need inventory management with barcode scanning and low stock alerts.",
        "As a student, I want an AI study assistant that summarizes textbooks and generates quiz questions."
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Example {i+1}: {example[:60]}...", key=f"ex_{i}"):
                st.session_state.last_example = example
                st.rerun()

# Load model only when needed
@st.cache_resource
def load_model():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, "./")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

# Generate output
if generate and requirement:
    with st.spinner("🤖 Generating user stories and modules..."):
        progress = st.progress(0)
        
        # Simulate progress
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        
        try:
            # Try to load model
            model, tokenizer = load_model()
            
            if model and tokenizer:
                # Prepare prompt
                prompt = f"""Convert this requirement to user stories and modules:
                
                Requirement: {requirement}
                
                User Stories:
                1."""
                
                # Generate
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display results
                st.success("✅ Generation complete!")
                
                # Split into sections
                if "User Stories:" in result and "Module Breakdown:" in result:
                    parts = result.split("Module Breakdown:")
                    stories = parts[0].replace("User Stories:", "").strip()
                    modules = parts[1].strip()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📖 User Stories")
                        st.markdown(stories)
                    
                    with col2:
                        st.markdown("### 🏗️ Module Breakdown")
                        st.markdown(modules)
                else:
                    st.markdown("### 📋 Generated Output")
                    st.markdown(result)
                
                # Download button
                st.download_button(
                    "📥 Download Results",
                    result,
                    file_name="user_stories.txt",
                    mime="text/plain"
                )
            else:
                # Fallback demo output
                st.warning("⚠️ Using demo mode (model not loaded)")
                
                st.markdown("### 📖 User Stories (Demo)")
                st.markdown("""
                1. **As a restaurant owner**, I can manage menu items with categories, prices, and descriptions
                2. **As a customer**, I can browse the menu, filter by category, and view item details
                3. **As a customer**, I can add items to cart, customize options, and place orders
                4. **As a restaurant**, I can receive and manage orders in real-time dashboard
                5. **As a customer**, I can track my order status and receive notifications
                """)
                
                st.markdown("### 🏗️ Module Breakdown (Demo)")
                st.markdown("""
                - **Menu Management Module**: CRUD operations for menu items
                - **Order Processing Module**: Handle orders from cart to kitchen
                - **Payment Integration Module**: Process payments securely
                - **Admin Dashboard Module**: Analytics and order management
                - **Notification Module**: SMS/Email alerts for order updates
                """)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            
            # Always show something
            st.markdown("### 📖 Sample User Stories")
            st.markdown(f"""
            1. **User Story 1**: Based on: {requirement[:50]}...
            2. **User Story 2**: Implementation of core functionality
            3. **User Story 3**: User authentication and authorization
            4. **User Story 4**: Data management and storage
            """)
            
            st.markdown("### 🏗️ Sample Modules")
            st.markdown("""
            - **Authentication Module**: User login/registration
            - **Core Function Module**: Main business logic
            - **Database Module**: Data storage and retrieval
            - **UI/UX Module**: User interface components
            - **API Module**: External integrations
            """)

# Footer
st.markdown("---")
st.caption("Powered by FLAN-T5 + LoRA | Files: " + ", ".join([f for f in required_files if os.path.exists(f)]))
