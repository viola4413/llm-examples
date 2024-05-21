import streamlit as st

from common_ui import page_setup
from conversation_manager import ConversationManager

user = st.session_state.get("user_name")
conv_mgr: ConversationManager = st.session_state.conversation_manager

page_setup(f"My Account: {user}", visibility="user")

conversations = conv_mgr.get_all_conversations(user=user)

metric_cols = st.columns(2)
metric_cols[0].metric("Total conversations x models", len(conversations))
metric_cols[1].metric("Total feedback", len([c for c in conversations if c.feedback]))

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
            st.write("✅ **Feedback submitted**" if c.feedback else "❌ **Feedback missing**")
            st.write(c.messages_to_text())
            for m in c.messages:
                if len(m.content) < 35:
                    txt = m.content
                else:
                    txt = m.content[0:35] + "..."
                f"**{m.role}:** {txt}"

    if st.button("Load conversation"):
        st.session_state.load_conversation = cr.title
        st.switch_page("app.py")
