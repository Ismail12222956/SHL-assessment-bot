import streamlit as st

st.set_page_config(page_title="SHL AI Assistant", page_icon="✨", layout="wide")

# ---------------------------------------------------
# ADVANCED PREMIUM THEME CSS
# ---------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Deep premium dark background */
    .stApp {
        background-color: #0B0E14; 
    }
    
    /* Top Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 0 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 600;
        background: linear-gradient(90deg, #FFFFFF, #A5B4FC, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #9CA3AF;
        font-size: 1.15rem;
        font-weight: 300;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Chat Input styling */
    .stChatInputContainer {
        border-radius: 24px !important;
        background: rgba(31, 41, 55, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
        padding: 0.6rem !important;
        margin-bottom: 2rem !important;
    }
    
    .stChatInputContainer textarea {
        color: #F3F4F6 !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chat message overrides */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 1rem !important;
        animation: slideUp 0.3s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom Avatars */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
        color: white;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: white;
    }
    
    /* Text colors inside chat */
    .stChatMessage p, .stChatMessage li {
        color: #E5E7EB !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .stChatMessage a {
        color: #818CF8 !important;
        text-decoration: none;
        transition: all 0.2s ease;
        border-bottom: 1px dashed rgba(129, 140, 248, 0.5);
    }
    
    .stChatMessage a:hover {
        color: #A5B4FC !important;
        border-bottom: 1px solid #A5B4FC;
    }
    
    .stChatMessage h3 {
        color: #A5B4FC !important;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✨ SHL Assessment AI</div>
    <div class="hero-subtitle">Intelligent candidate matching & assessment recommendations</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# RENDER EXISTING MESSAGES
# ---------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# CHAT INPUT & LOGIC
# ---------------------------------------------------
if prompt := st.chat_input("Ask about candidate assessments... (e.g. 'I need a coding test')"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analyzing requirements... ⏳")
        
        try:
            from app.chatbot import process_chat
            from app.models import Message
            
            # Convert st.session_state.messages to List[Message]
            pydantic_messages = [Message(role=m["role"], content=m["content"]) for m in st.session_state.messages]
            
            # Call process_chat directly
            data = process_chat(pydantic_messages)
            
            assistant_reply = data.get("reply", "")
            final_response = assistant_reply
            
            message_placeholder.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
        except Exception as e:
            error_msg = f"⚠️ Unexpected Error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})