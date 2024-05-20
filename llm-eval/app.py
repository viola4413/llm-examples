from typing import List
import streamlit as st

from common_ui import (
    chat_response,
    configure_model,
    page_setup,
    st_thread,
)
from schema import Conversation, Message

page_setup(
    title="Tasty Bytes Chat",
    icon="❄️",
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
feedback_controls = st.empty()
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
    st.toast("Feedback submitted!", icon=":material/rate_review:")


def clear_history():
    for conversation in conversations:
        conversation.reset_messages()
        conversation.add_message(Message(role="assistant", content=DEFAULT_MESSAGE), render=False)


def regenerate():
    st.session_state.regenerate = conversations[0].messages[-2].content
    for conversation in conversations:
        del conversation.messages[-2:]


if len(conversations[0].messages) > 1:
    feedback_cols = feedback_controls.columns(4)

    BUTTON_LABELS = [
        "👈&nbsp; wins",
        "👉&nbsp; wins",
        "🤝&nbsp; Tie",
        "👎&nbsp; Both bad",
    ]
    for i, label in enumerate(BUTTON_LABELS):
        with feedback_cols[i]:
            st.button(
                label,
                use_container_width=True,
                on_click=record_feedback,
            )

    # TODO: Big loading skeleton always briefly shows on the hosted app
    action_cols = response_controls.columns(3)

    action_cols[0].button("🔄&nbsp; Regenerate", use_container_width=True, on_click=regenerate)
    action_cols[1].button(
        "🗑&nbsp; Clear history",
        use_container_width=True,
        on_click=clear_history,
    )
