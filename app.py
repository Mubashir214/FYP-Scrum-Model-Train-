import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import time
import os
from typing import Optional, Dict, List
import json

# ============================================
# 1. Page Configuration
# ============================================
st.set_page_config(
    page_title="AI User Stories Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 2. Custom CSS Styling
# ============================================
st.markdown("""
<style>
    /* Main container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 800;
        padding: 10px;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    /* Result sections */
    .result-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-left: 6px solid;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .stories-section {
        border-left-color: #10b981;
    }
    
    .modules-section {
        border-left-color: #8b5cf6;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .stories-header {
        color: #10b981;
    }
    
    .modules-header {
        color: #8b5cf6;
    }
    
    /* Bullet points */
    .story-item {
        background: white;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    
    .story-icon {
        font-size: 1.2rem;
        min-width: 30px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .generate-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        font-size: 1rem;
        padding: 15px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    /* Example box */
    .example-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #f5f0ff 100%);
        border: 2px dashed #c7d2fe;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-box:hover {
        background: linear-gradient(135deg, #e0e7ff 0%, #ede9fe 100%);
        transform: translateX(5px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .card {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 3. Model Loading Class (Cached)
# ============================================
class UserStoriesModel:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.adapter_path = "finetuned_lora"
        self.loaded = False
        
    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Load the fine-tuned model with caching"""
        try:
            # Check if adapter files exist
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            for file in required_files:
                if not os.path.exists(os.path.join(_self.adapter_path, file)):
                    st.error(f"Missing required file: {file}")
                    return None, None
            
            # Configure 4-bit quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load base model
            with st.spinner("🔄 Loading base model (FLAN-T5)..."):
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "google/flan-t5-base",
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
            
            # Load fine-tuned adapter
            with st.spinner("🔄 Loading fine-tuned adapter..."):
                model = PeftModel.from_pretrained(
                    base_model,
                    _self.adapter_path,
                    adapter_name="user_stories_adapter"
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(_self.adapter_path)
            
            _self.loaded = True
            return model, tokenizer
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    def generate(self, requirement: str, temperature: float = 0.7, max_length: int = 512) -> str:
        """Generate user stories and module breakdown"""
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = self.load_model()
        
        if self.model is None:
            return "Error: Model failed to load"
        
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
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            # Decode output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            return f"Error during generation: {str(e)}"

# ============================================
# 4. Helper Functions
# ============================================
def parse_output(result_text: str) -> Dict[str, List[str]]:
    """Parse the model output into structured format"""
    stories = []
    modules = []
    
    lines = result_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'user stories' in line.lower() or 'user stories:' in line:
            current_section = 'stories'
        elif 'module breakdown' in line.lower() or 'module breakdown:' in line:
            current_section = 'modules'
        elif current_section == 'stories':
            # Clean up bullet points and numbering
            clean_line = line.strip('•-1234567890. ')
            if clean_line and len(clean_line) > 5:
                stories.append(clean_line)
        elif current_section == 'modules':
            clean_line = line.strip('•- ')
            if clean_line and len(clean_line) > 5:
                modules.append(clean_line)
    
    return {
        "user_stories": stories[:10],  # Limit to 10 items
        "module_breakdown": modules[:10],
        "raw_output": result_text
    }

def create_download_content(parsed_result: Dict) -> str:
    """Create downloadable content"""
    content = "=" * 60 + "\n"
    content += "AI-GENERATED USER STORIES & MODULE BREAKDOWN\n"
    content += "=" * 60 + "\n\n"
    
    content += "USER STORIES:\n"
    content += "-" * 40 + "\n"
    for i, story in enumerate(parsed_result["user_stories"], 1):
        content += f"{i}. {story}\n"
    
    content += "\n\nMODULE BREAKDOWN:\n"
    content += "-" * 40 + "\n"
    for i, module in enumerate(parsed_result["module_breakdown"], 1):
        content += f"{i}. {module}\n"
    
    content += "\n\nRAW OUTPUT:\n"
    content += "-" * 40 + "\n"
    content += parsed_result["raw_output"]
    
    return content

# ============================================
# 5. Streamlit App Layout
# ============================================
def main():
    # Initialize model
    model_manager = UserStoriesModel.get_instance()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 AI User Stories & Module Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Generation Settings")
        
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative, Lower values = more focused"
        )
        
        max_length = st.slider(
            "Output Length",
            min_value=256,
            max_value=1024,
            value=512,
            step=128,
            help="Maximum length of generated text"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Status")
        
        if model_manager.loaded:
            st.success("✅ Model Loaded Successfully")
        else:
            st.info("🔄 Model will load on first generation")
        
        st.markdown("**Model:** FLAN-T5 Base")
        st.markdown("**Fine-tuning:** LoRA Adapter")
        st.markdown("**Task:** Requirements → User Stories")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 Tips")
        st.info("""
        1. Be specific in your requirements
        2. Include user roles if possible
        3. Mention key features needed
        4. Specify technical constraints
        5. Add business objectives
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main Content Area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter User Requirement")
        
        requirement = st.text_area(
            "Describe your system or feature requirement:",
            height=200,
            placeholder="Example: As a project manager, I need a dashboard to track team progress, deadlines, and resource allocation in real-time with automated reporting...",
            key="requirement_input"
        )
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            generate_clicked = st.button(
                "🚀 Generate User Stories",
                type="primary",
                use_container_width=True,
                key="generate_btn"
            )
        with col1_2:
            clear_clicked = st.button(
                "🗑️ Clear",
                use_container_width=True,
                key="clear_btn"
            )
        
        if clear_clicked:
            st.session_state.clear()
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Example Requirements
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 Example Requirements")
        
        examples = [
            "As a restaurant owner, I want a mobile app for customers to view menu, place orders, make payments, and track delivery status in real-time.",
            "As a fitness coach, I need a platform to create personalized workout plans, track client progress, schedule sessions, and share nutrition guides.",
            "As an e-commerce business, we need an inventory management system with barcode scanning, low stock alerts, supplier management, and sales analytics.",
            "As a student, I want an AI-powered study assistant that can summarize textbooks, generate quiz questions, track study time, and provide progress reports.",
            "As a hospital administrator, we need a patient management system for appointment scheduling, medical records, prescription tracking, and insurance billing."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"📋 Example {i}", key=f"example_{i}", use_container_width=True):
                st.session_state.requirement_input = example
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if generate_clicked and requirement:
            with st.spinner("🤖 Generating user stories and modules..."):
                start_time = time.time()
                
                # Generate output
                result = model_manager.generate(requirement, temperature, max_length)
                processing_time = time.time() - start_time
                
                # Parse result
                parsed = parse_output(result)
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("⏱️ Time", f"{processing_time:.2f}s")
                with metric_col2:
                    st.metric("📖 Stories", len(parsed["user_stories"]))
                with metric_col3:
                    st.metric("🏗️ Modules", len(parsed["module_breakdown"]))
                
                # Display User Stories
                if parsed["user_stories"]:
                    st.markdown('<div class="result-section stories-section">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header stories-header">📖 USER STORIES</div>', unsafe_allow_html=True)
                    
                    for i, story in enumerate(parsed["user_stories"], 1):
                        st.markdown(f"""
                        <div class="story-item">
                            <div class="story-icon">🎯</div>
                            <div>
                                <strong>Story {i}:</strong> {story}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display Module Breakdown
                if parsed["module_breakdown"]:
                    st.markdown('<div class="result-section modules-section">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header modules-header">🏗️ MODULE BREAKDOWN</div>', unsafe_allow_html=True)
                    
                    for i, module in enumerate(parsed["module_breakdown"], 1):
                        st.markdown(f"""
                        <div class="story-item">
                            <div class="story-icon">🔧</div>
                            <div>
                                <strong>Module {i}:</strong> {module}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Download button
                download_content = create_download_content(parsed)
                st.download_button(
                    label="📥 Download Full Report",
                    data=download_content,
                    file_name="user_stories_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Raw output expander
                with st.expander("📋 View Raw Output"):
                    st.code(result, language="text")
        
        elif not requirement and generate_clicked:
            st.warning("⚠️ Please enter a requirement first!")
        
        else:
            # Placeholder when no generation has been done
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Generated Output")
            st.info("""
            Your generated user stories and module breakdown will appear here.
            
            **What to expect:**
            - Detailed user stories with acceptance criteria
            - Technical module breakdown
            - Feature prioritization
            - Implementation suggestions
            
            Click **'Generate User Stories'** to start!
            """)
            
            # Quick preview
            st.markdown("#### 🎯 Sample Output Preview")
            st.markdown("""
            **User Stories:**
            1. As a user, I can create an account with email verification
            2. As a user, I can upload and preview PDF files
            3. As a user, I can share files with team members
            
            **Module Breakdown:**
            - Authentication Module
            - File Upload Service
            - Sharing & Permissions
            - Notification System
            """)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <hr>
        <p>🚀 Powered by Fine-tuned FLAN-T5 with LoRA | Made with Streamlit</p>
        <p>📚 Model: FLAN-T5-Base + Custom Adapter | 🏗️ Fine-tuned on User Requirements Dataset</p>
        <p>⚠️ AI-generated content should be reviewed by domain experts</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# 6. Run the App
# ============================================
if __name__ == "__main__":
    # Check for required files
    required_files = [
        "finetuned_lora/adapter_config.json",
        "finetuned_lora/adapter_model.safetensors",
        "finetuned_lora/tokenizer_config.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure all model files are in the 'finetuned_lora' directory")
    else:
        main()
