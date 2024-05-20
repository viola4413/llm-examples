from typing import List
import streamlit as st

from common_ui import (
    chat_response,
    configure_model,
    page_setup,
    st_thread,
)
from schema import (
    Conversation,
    ConversationRecord,
    Message,
)

title = "Chat"
if st.session_state.get("conversation_title"):
    title = f"Chat: {st.session_state.conversation_title}"

page_setup(
    title=title,
    wide_mode=True,
    collapse_sidebar=False,
)

DEFAULT_MESSAGE = "Hello there! Let's chat?"
MODELS_HELP_STR = "Select an available model"
AVAILABLE_MODELS = [
    "snowflake/snowflake-arctic-instruct",
    "meta/meta-llama-3-8b",
    "mistralai/mistral-7b-instruct-v0.2",
]

# Store conversation state in streamlit session
if "conversations" not in st.session_state:
    st.session_state["conversations"] = [Conversation(), Conversation()]
    for conversation in st.session_state["conversations"]:
        conversation.add_message(Message(role="assistant", content=DEFAULT_MESSAGE), render=False)
conversations: List[Conversation] = st.session_state["conversations"]

# Handle case where we navigated to load an existing conversation:
if to_load := st.session_state.pop("load_conversation", None):
    index = [c.title for c in st.session_state.chat_history].index(to_load)
    cr = st.session_state.chat_history[index]
    st.session_state["conversations"] = cr.conversations
    st.session_state["conversation_title"] = cr.title
    st.rerun()

# Main area

""

model_cols = st.columns(len(conversations))
for idx, conversation in enumerate(conversations):
    conversation.model_config = configure_model(model_cols[idx], conversation.model_config, key=f"{idx}")

# Render the chat
for idx, msg in enumerate(conversations[0].messages):
    if msg.role == "user":
        conversations[0].render_message(msg)
    else:
        msg_cols = st.columns(len(conversations))
        for i, conversation in enumerate(conversations):
            conversation.render_message(
                conversation.messages[idx],
                container=msg_cols[i],
            )

user_msg = st.empty()
response = st.empty()
response_controls = st.empty()

user_input = st.chat_input("Enter your message here.") or st.session_state.pop("regenerate", None)
if user_input:
    new_msg = Message(role="user", content=user_input)
    for c in conversations:
        c.add_message(new_msg, render=False)
    conversations[0].render_message(new_msg, container=user_msg)

    msg_cols = response.columns(len(conversations))
    threads = []
    for i, conversation in enumerate(conversations):
        args = (
            conversation,
            msg_cols[i],
        )
        t = st_thread(target=chat_response, args=args)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    st.rerun()  # Clear stale containers

# Add action buttons


def record_feedback():
    # TODO: This should also persist the ConversationRecord in session state
    st.toast("Feedback submitted!", icon=":material/rate_review:")


def clear_history():
    for conversation in conversations:
        conversation.reset_messages()
        conversation.add_message(Message(role="assistant", content=DEFAULT_MESSAGE), render=False)


def regenerate():
    st.session_state.regenerate = conversations[0].messages[-2].content
    for conversation in conversations:
        del conversation.messages[-2:]


@st.experimental_dialog("Edit conversation title")
def edit_title():
    new_title = st.text_input(
        "New conversation title:",
        value=st.session_state.get("conversation_title"),
    )
    if st.button("Save"):
        st.session_state.conversation_title = new_title
        st.rerun()


if len(conversations[0].messages) > 1:
    action_cols = response_controls.columns(3)

    action_cols[0].button("🔄&nbsp; Regenerate", use_container_width=True, on_click=regenerate)
    action_cols[1].button(
        "🗑&nbsp; Clear history",
        use_container_width=True,
        on_click=clear_history,
    )

    with st.sidebar:
        if st.button("✏️&nbsp; Edit title", use_container_width=True):
            edit_title()
        if st.button("Export", use_container_width=True):
            cr = ConversationRecord()
            cr.user = st.session_state.get("user_name")
            cr.conversations = conversations
            cr.title = st.session_state.get("conversation_title")
            print(cr.to_json())
