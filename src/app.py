from model import MedGemmaEngine
import streamlit as st
from database import save_chat_turn, get_all_history

st.set_page_config(page_title="Med-Gemma AI Assistant", page_icon="🏥")
st.title("🏥 Med-Gemma Clinical Assistant")

# 2. Cache the Model Initialization
@st.cache_resource
def get_model():
    manager = MedGemmaEngine()
    manager.initialize()
    return manager

# Initialize the engine
with st.spinner("Waking up the medical brain... this may take a moment."):
    engine = get_model()

#Sidebar to access history
with st.sidebar:
    st.title("🏥 Clinical History")
    if st.button("➕ New Consultation"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("Previous Notes")
    # Fetch list of past sessions from NoSQL
    past_sessions = get_all_history()
    for session in past_sessions[-5:]:  # Show last 5
        st.caption(f"📅 {session['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        st.text(session['user_input'][:30] + "...")
        st.text(session['ai_output'][:30]+"....")

# Load history on startup
if "messages" not in st.session_state:
    raw_history = get_all_history()
    # Flatten history for the UI
    st.session_state.messages = []
    for doc in raw_history:
        st.session_state.messages.append({"role": "user", "content": doc["user_input"]})
        st.session_state.messages.append({"role": "assistant", "content": doc["ai_output"]})

if prompt := st.chat_input("Enter query..."):
    # Display user input
    st.chat_message("user").markdown(prompt)
    
    # Generate response
    engine = get_model()
    response = engine.generate_response(prompt)
    
    # Display assistant response
    st.chat_message("assistant").markdown(response)
    
    # SAVE TO NoSQL: Note how easy it is to add extra data
    save_chat_turn(
        user_query=prompt, 
        ai_response=response,
        metadata={"tokens_used": len(response.split())} # Extra info!
    )
