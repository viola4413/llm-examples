import streamlit as st

from common_ui import page_setup
from conversation_manager import ConversationManager

page_setup("Personal Stats", visibility="user")

user = st.session_state.get("user_name")
conv_mgr: ConversationManager = st.session_state.conversation_manager

st.header("Conversation history")

options = [""] + conv_mgr.list_conversations_by_user(user)
selected = st.selectbox("Select a conversation:", options)
if selected:
    cr = conv_mgr.get_by_title(selected)
    st.subheader(cr.title)
    cols = st.columns(len(cr.conversations))
    for idx, col in enumerate(cols):
        with col:
            c = cr.conversations[idx]
            "**Model Config**"
            st.json(dict(c.model_config), expanded=False)
            for m in c.messages:
                if len(m.content) < 35:
                    txt = m.content
                else:
                    txt = m.content[0:35] + "..."
                f"**{m.role}:** {txt}"

    if st.button("Load conversation"):
        st.session_state.load_conversation = cr.title
        st.switch_page("app.py")
