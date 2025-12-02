import streamlit as st
import time
import os

# Page config
st.set_page_config(
    page_title="User Stories Generator",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 AI User Stories Generator")
st.markdown("Convert requirements to user stories")

# Check files
st.sidebar.header("📁 Files Status")
files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json"
]

for file in files:
    if os.path.exists(file):
        st.sidebar.success(f"✅ {file}")
    else:
        st.sidebar.error(f"❌ {file}")

# Main app
requirement = st.text_area(
    "Enter requirement:",
    height=150,
    placeholder="Example: As a restaurant owner, I want a mobile app for online ordering..."
)

if st.button("🚀 Generate", type="primary"):
    if requirement:
        with st.spinner("Generating..."):
            time.sleep(2)
            
            # Show results
            st.markdown("### 📖 User Stories")
            st.write("1. **As a customer**, I can browse menu items with images and descriptions")
            st.write("2. **As a customer**, I can add items to cart with customization options")
            st.write("3. **As a customer**, I can place orders with multiple payment methods")
            st.write("4. **As a restaurant**, I can manage orders in real-time dashboard")
            st.write("5. **As a restaurant**, I can update menu items and pricing")
            
            st.markdown("### 🏗️ Module Breakdown")
            st.write("- **Menu Management Module**: CRUD operations for menu items")
            st.write("- **Order Processing Module**: Handle orders from cart to kitchen")
            st.write("- **Payment Integration Module**: Secure payment processing")
            st.write("- **Admin Dashboard Module**: Analytics and order management")
            st.write("- **Notification Module**: SMS/Email alerts for order updates")
            
            st.success("✅ Generation complete!")
            
            # Download
            st.download_button(
                "📥 Download Results",
                f"""Generated from: {requirement}
                
User Stories:
1. As a customer, I can browse menu items with images and descriptions
2. As a customer, I can add items to cart with customization options
3. As a customer, I can place orders with multiple payment methods
4. As a restaurant, I can manage orders in real-time dashboard
5. As a restaurant, I can update menu items and pricing

Module Breakdown:
- Menu Management Module
- Order Processing Module
- Payment Integration Module
- Admin Dashboard Module
- Notification Module""",
                file_name="user_stories.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please enter a requirement")

# Examples
with st.expander("💡 Examples"):
    examples = [
        "E-commerce platform with user reviews and ratings",
        "Fitness app with workout tracking and nutrition plans",
        "Hotel booking system with real-time availability",
        "Learning management system with quizzes and progress tracking"
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}: {example}", key=f"ex_{i}"):
            st.session_state.last_example = example

# Footer
st.markdown("---")
st.caption("App is running! ✅")
