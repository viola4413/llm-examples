import streamlit as st

from common_ui import page_setup

page_setup("Personal Stats", visibility="user")

user = st.session_state.get("user_name")
history = [c for c in st.session_state.chat_history if c.user == user]

st.header("Conversation history")

selected = st.selectbox("Select a conversation:", [""] + [c.title for c in history])
if selected:
    index = [c.title for c in history].index(selected)
    cr = history[index]
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
