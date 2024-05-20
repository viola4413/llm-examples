import pathlib
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from conversation_manager import ConversationManager
from llm import generate_stream
from schema import (
    Conversation,
    Message,
    ModelConfig,
)


AVAILABLE_MODELS = [
    "snowflake/snowflake-arctic-instruct",
    "meta/meta-llama-3-8b",
    "mistralai/mistral-7b-instruct-v0.2",
]


def page_setup(title, wide_mode=False, collapse_sidebar=False, visibility="public"):
    if "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.rerun()

    # Handle access control
    if visibility in ("user", "admin") and not st.session_state.get("user_name"):
        st.switch_page("app.py")
    if visibility == "admin" and not st.session_state.get("admin_mode"):
        st.switch_page("app.py")

    CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
    LOGO = str(CURRENT_DIR / "logo.png")
    ICON_LOGO = str(CURRENT_DIR / "logo_small.png")

    st.set_page_config(
        page_title=f"LLM Evaluation: {title}",
        page_icon=ICON_LOGO,
        layout="wide" if wide_mode else "centered",
        initial_sidebar_state="collapsed" if collapse_sidebar else "auto",
    )

    st.logo(LOGO, link="https://www.snowflake.com", icon_image=ICON_LOGO)
    st.title(title)

    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    # Add page navigation
    with st.sidebar:
        st.header("LLM Evaluation")

        st.write("")

        st.page_link("app.py", label="Chat", icon=":material/chat:")
        st.page_link("pages/about.py", label="About", icon=":material/info:")

        if st.session_state.get("user_name"):
            st.page_link("pages/personal.py", label="Personal Stats", icon=":material/star:")

        if st.session_state.get("admin_mode"):
            st.subheader("Admin view")
            st.page_link("pages/analysis.py", label="Conversation Analysis", icon=":material/analytics:")
            st.page_link("pages/auto_eval.py", label="Automated Evaluation", icon=":material/quiz:")
            st.page_link("pages/users.py", label="User Management", icon=":material/group:")

        st.write("")
        st.write("")

        if not st.session_state.get("user_name"):
            if st.button("🔑&nbsp; Login", use_container_width=True):
                login()
        else:
            st.write(f"Logged in user: `{st.session_state.user_name}`")
            if st.button("🔑&nbsp; Logout", use_container_width=True):
                st.session_state.user_name = None
                st.session_state.admin_mode = None
                if visibility != "public":
                    st.switch_page("app.py")
                else:
                    st.rerun()


@st.experimental_dialog("Login")
def login():
    user_name = st.text_input("Username:")
    admin_mode = st.checkbox("Admin mode")
    if st.button("Submit"):
        st.session_state.user_name = user_name
        st.session_state.admin_mode = admin_mode
        st.rerun()


def configure_model(container, model_config: ModelConfig, key: str):
    with container:
        with st.popover(f"Configure `{model_config.model}`", use_container_width=True):
            MODEL_KEY = f"model_{key}"
            TEMPERATURE_KEY = f"temperature_{key}"
            TOP_P_KEY = f"top_p_{key}"
            MAX_NEW_TOKENS_KEY = f"max_new_tokens_{key}"

            if MODEL_KEY not in st.session_state:
                st.session_state[MODEL_KEY] = model_config.model
                st.session_state[TEMPERATURE_KEY] = model_config.temperature
                st.session_state[TOP_P_KEY] = model_config.top_p
                st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens

            model_config.model = st.selectbox(
                label="Select model:",
                options=AVAILABLE_MODELS,
                key=MODEL_KEY,
            )

            model_config.temperature = st.slider(
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                label="Temperature:",
                key=TEMPERATURE_KEY,
            )

            model_config.top_p = st.slider(
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                label="Top P:",
                key=TOP_P_KEY,
            )

            model_config.max_new_tokens = st.slider(
                min_value=100,
                max_value=1500,
                step=100,
                label="Max new tokens:",
                key=MAX_NEW_TOKENS_KEY,
            )
    return model_config


def chat_response(
    conversation: Conversation,
    container=None,
):
    stream_iter = generate_stream(conversation)

    if container:
        chat = container.chat_message("assistant")
    else:
        chat = st.chat_message("assistant")
    full_streamed_response = chat.write_stream(stream_iter)
    conversation.add_message(
        Message(role="assistant", content=str(full_streamed_response).strip()),
        render=False,
    )


def st_thread(target, args) -> threading.Thread:
    """Return a function as a Streamlit-safe thread"""

    thread = threading.Thread(target=target, args=args)
    add_script_run_ctx(thread, get_script_run_ctx())
    return thread
