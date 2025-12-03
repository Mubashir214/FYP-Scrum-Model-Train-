import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
import time

# Set page config
st.set_page_config(
    page_title="User Requirements to User Stories Converter",
    page_icon="📋",
    layout="wide"
)

# Disable torch.compile globally
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

@st.cache_resource
def load_model():
    """Load the fine-tuned model once and cache it"""
    try:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        base_model = "google/flan-t5-base"
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load your trained LoRA adapter
        model = PeftModel.from_pretrained(model, "./")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_response(user_requirement, model, tokenizer):
    """Generate user stories and module breakdown from requirement"""
    instruction = "Convert the following user requirement into detailed User Stories and a Module Breakdown."
    
    prompt = f"""{instruction}

User Requirement: {user_requirement.strip()}

"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            min_length=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.1,
            num_beams=1,
            early_stopping=True
        )
    
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def format_response(response):
    """Format the response with proper Markdown styling"""
    formatted = ""
    
    # Split by sections
    if "User Stories:" in response and "Module Breakdown:" in response:
        user_stories, module_breakdown = response.split("Module Breakdown:")
        user_stories = user_stories.replace("User Stories:", "").strip()
        module_breakdown = module_breakdown.strip()
        
        formatted += "### 📝 User Stories\n"
        formatted += "```\n"
        formatted += user_stories + "\n"
        formatted += "```\n\n"
        
        formatted += "### 🏗️ Module Breakdown\n"
        formatted += "```\n"
        formatted += module_breakdown + "\n"
        formatted += "```"
    else:
        # Fallback formatting
        formatted = response.replace("User Stories:", "### 📝 User Stories\n```\n").replace("Module Breakdown:", "```\n\n### 🏗️ Module Breakdown\n```\n") + "\n```"
    
    return formatted

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 20px;
        background-color: #f0f8ff;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📋 User Requirements to User Stories Converter")
    st.markdown("Transform user requirements into detailed **User Stories** and **Module Breakdown** using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses a fine-tuned FLAN-T5 model to:
        - Convert user requirements into actionable user stories
        - Generate comprehensive module breakdowns
        - Help with software planning and development
        
        **How to use:**
        1. Enter your user requirement
        2. Click 'Generate'
        3. Copy the formatted output
        """)
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Base Model:** FLAN-T5-Base
        - **Fine-tuning:** LoRA + 4-bit quantization
        - **Purpose:** Requirement analysis
        """)
    
    # Load model with progress
    with st.spinner("Loading AI model... This may take a minute on first run."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check if model files are in the correct directory.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter User Requirement")
        user_input = st.text_area(
            "Describe your user requirement in detail:",
            placeholder="Example: As a user, I want to be able to upload images and get automatic tags so that I can organize my photo collection more efficiently.",
            height=200,
            key="user_input"
        )
        
        generate_button = st.button("🚀 Generate User Stories & Module Breakdown", type="primary")
    
    with col2:
        st.subheader("📚 Example Requirements")
        examples = [
            "As a parent, I want to receive push notifications when my child's school bus is 5 minutes away so that I can be ready at the stop.",
            "As a freelancer, I want to create and send professional invoices directly from the app with automatic tax calculation and payment reminders.",
            "Take photo of houseplant and get instant care tips, watering reminders, and disease detection.",
            "Scan any product barcode in store and instantly see if it's cheaper online with price comparison and reviews."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}: {example[:80]}...", key=f"example_{i}"):
                st.session_state.user_input = example
                st.rerun()
    
    # Generation section
    if generate_button and user_input:
        with st.spinner("Analyzing requirement and generating detailed output..."):
            start_time = time.time()
            
            # Generate response
            response = generate_response(user_input, model, tokenizer)
            
            elapsed_time = time.time() - start_time
            
            # Display results
            st.markdown("---")
            st.subheader("✅ Generated Output")
            st.caption(f"Generated in {elapsed_time:.2f} seconds")
            
            # Display in expandable sections
            formatted_response = format_response(response)
            
            # Raw response expander
            with st.expander("📄 View Raw Output", expanded=False):
                st.code(response, language="text")
            
            # Formatted output
            st.markdown(formatted_response, unsafe_allow_html=True)
            
            # Copy button
            st.markdown("---")
            col_copy1, col_copy2 = st.columns(2)
            with col_copy1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(response, language="text")
                    st.success("Output copied to clipboard! (Manual copy needed)")
            with col_copy2:
                st.download_button(
                    label="💾 Download as TXT",
                    data=response,
                    file_name="user_stories_output.txt",
                    mime="text/plain"
                )
    
    elif generate_button and not user_input:
        st.warning("⚠️ Please enter a user requirement first!")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & Hugging Face Transformers")

if __name__ == "__main__":
    main()
