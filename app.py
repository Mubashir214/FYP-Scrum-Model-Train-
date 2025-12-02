import streamlit as st
import torch
import os
import time
import sys
from pathlib import Path

# ============================================
# 1. Check and Install Missing Dependencies
# ============================================
def check_and_install_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "safetensors",
        "protobuf"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        st.warning(f"⚠️ Missing packages: {', '.join(missing)}")
        st.info("Please add these to your requirements.txt and redeploy")
        return False
    return True

# ============================================
# 2. Page Configuration
# ============================================
st.set_page_config(
    page_title="AI User Stories Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 3. Custom CSS (Inline for Streamlit Cloud)
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .result-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .stories-header {
        color: #10B981;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .modules-header {
        color: #8B5CF6;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    .example-box {
        background: #F0F4FF;
        border: 2px dashed #C7D2FE;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .example-box:hover {
        background: #E0E7FF;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 4. Model Loading with Error Handling
# ============================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the model with comprehensive error handling"""
    try:
        # Check if model files exist
        model_dir = Path("finetuned_lora")
        required_files = ['adapter_config.json', 'adapter_model.safetensors']
        
        for file in required_files:
            if not (model_dir / file).exists():
                st.error(f"❌ Missing file: {model_dir / file}")
                st.info("Please make sure all model files are uploaded to Streamlit Cloud")
                return None, None
        
        # Import here to avoid issues if dependencies are missing
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        # Configure for CPU/GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load with 4-bit quantization if GPU available, otherwise CPU
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            # CPU mode (for Streamlit Cloud free tier)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # Load fine-tuned adapter
        model = PeftModel.from_pretrained(
            base_model,
            str(model_dir),
            adapter_name="user_stories_adapter"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        return model, tokenizer
        
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
        st.info("Please check your requirements.txt file")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================
# 5. Helper Functions
# ============================================
def generate_output(model, tokenizer, requirement, temperature=0.7, max_length=512):
    """Generate user stories and module breakdown"""
    try:
        # Create prompt
        prompt = f"""Convert the following user requirement into detailed User Stories and a Module Breakdown.

User Requirement: {requirement.strip()}

Output format:
User Stories:
1. [First user story]
2. [Second user story]

Module Breakdown:
- [Module 1]
- [Module 2]"""
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
        
        # Decode output
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
        
    except Exception as e:
        return f"Error during generation: {str(e)}"

def parse_output(result_text):
    """Parse the model output into structured format"""
    stories = []
    modules = []
    
    lines = result_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'user stories' in line.lower():
            current_section = 'stories'
        elif 'module breakdown' in line.lower():
            current_section = 'modules'
        elif current_section == 'stories':
            # Clean up bullet points and numbering
            if line and len(line) > 5 and not line.startswith('Module'):
                stories.append(line)
        elif current_section == 'modules':
            if line and len(line) > 5:
                modules.append(line)
    
    return {
        "user_stories": stories[:8],
        "module_breakdown": modules[:8],
        "raw_output": result_text
    }

# ============================================
# 6. Main App
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 AI User Stories Generator</h1>', unsafe_allow_html=True)
    
    # Check dependencies
    if not check_and_install_dependencies():
        st.stop()
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'requirement' not in st.session_state:
        st.session_state.requirement = ""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        max_length = st.slider(
            "Max Output Length",
            min_value=256,
            max_value=768,
            value=512,
            step=128
        )
        
        st.header("ℹ️ About")
        st.info("""
        This tool converts user requirements into:
        - 📖 Detailed user stories
        - 🏗️ Module breakdown
        - 🔧 Technical specifications
        
        Powered by fine-tuned FLAN-T5 model.
        """)
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter User Requirement")
        
        requirement = st.text_area(
            "Describe your requirement:",
            height=150,
            placeholder="Example: As a project manager, I need a dashboard to track team progress and deadlines...",
            key="input_requirement"
        )
        
        if st.button("🚀 Generate", type="primary", use_container_width=True):
            if requirement.strip():
                st.session_state.requirement = requirement.strip()
            else:
                st.warning("Please enter a requirement first!")
    
    with col2:
        st.subheader("💡 Examples")
        
        examples = [
            "As a restaurant owner, I want a mobile app for customers to view menu, place orders, and track delivery.",
            "As a fitness coach, I need a platform to create workout plans and track client progress.",
            "As an e-commerce business, we need inventory management with barcode scanning and alerts.",
            "As a student, I want an AI study assistant that can summarize textbooks and generate quizzes."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}: {example[:50]}...", key=f"ex_{i}"):
                st.session_state.input_requirement = example
                st.rerun()
    
    # Generate Output
    if st.session_state.requirement:
        st.divider()
        
        with st.spinner("🤖 Generating user stories and modules..."):
            # Load model if not loaded
            if st.session_state.model is None:
                model, tokenizer = load_model()
                if model is None:
                    st.error("Failed to load model. Please check your model files.")
                    st.stop()
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
            
            # Generate
            start_time = time.time()
            result = generate_output(
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.requirement,
                temperature,
                max_length
            )
            processing_time = time.time() - start_time
            
            # Parse and display
            parsed = parse_output(result)
            
            # Metrics
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("⏱️ Time", f"{processing_time:.2f}s")
            with col_metrics[1]:
                st.metric("📖 Stories", len(parsed["user_stories"]))
            with col_metrics[2]:
                st.metric("🏗️ Modules", len(parsed["module_breakdown"]))
            
            # Display results
            if parsed["user_stories"]:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="stories-header">📖 USER STORIES</div>', unsafe_allow_html=True)
                for i, story in enumerate(parsed["user_stories"], 1):
                    st.markdown(f"**{i}.** {story}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if parsed["module_breakdown"]:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<div class="modules-header">🏗️ MODULE BREAKDOWN</div>', unsafe_allow_html=True)
                for i, module in enumerate(parsed["module_breakdown"], 1):
                    st.markdown(f"**{i}.** {module}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            download_content = f"""User Stories & Module Breakdown
Generated from: {st.session_state.requirement}

USER STORIES:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(parsed['user_stories'])])}

MODULE BREAKDOWN:
{chr(10).join([f"{i+1}. {m}" for i, m in enumerate(parsed['module_breakdown'])])}

Generated in {processing_time:.2f} seconds"""
            
            st.download_button(
                label="📥 Download Results",
                data=download_content,
                file_name="user_stories.txt",
                mime="text/plain"
            )
            
            # Show raw output in expander
            with st.expander("📋 View Raw Output"):
                st.code(result)
    
    # Footer
    st.divider()
    st.caption("🚀 Powered by Fine-tuned FLAN-T5 | Made with Streamlit")

# ============================================
# 7. Run the App
# ============================================
if __name__ == "__main__":
    # Check if model directory exists
    if not os.path.exists("finetuned_lora"):
        st.error("❌ 'finetuned_lora' directory not found!")
        st.info("""
        Please upload your model files to Streamlit Cloud:
        1. Create a folder named 'finetuned_lora' in your app directory
        2. Upload all your model files:
           - adapter_config.json
           - adapter_model.safetensors
           - tokenizer_config.json
           - tokenizer.json
           - special_tokens_map.json
        3. Redeploy your app
        """)
        st.stop()
    
    main()
