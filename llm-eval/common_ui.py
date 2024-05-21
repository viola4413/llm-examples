import pathlib
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from conversation_manager import ConversationManager
from llm import generate_stream, AVAILABLE_MODELS
from schema import (
    Conversation,
    Message,
    ModelConfig,
)


def page_setup(title, wide_mode=False, collapse_sidebar=False, visibility="public"):
    if st.get_option("client.showSidebarNavigation") and "already_ran" not in st.session_state:
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

    # Check for initial login via query_params
    if initial_user := st.query_params.get("user"):
        st.session_state.user_name = initial_user
        del st.query_params["user"]

    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()

    # Add page navigation
    with st.sidebar:
        st.header("LLM Evaluation")

        # st.write("")

        st.page_link("pages/about.py", label="About", icon=":material/info:")
        st.page_link("app.py", label="Chat", icon=":material/chat:")

        if st.session_state.get("user_name"):
            st.page_link("pages/account.py", label="My Account", icon=":material/account_circle:")

        if st.session_state.get("admin_mode"):
            st.subheader("Admin view")
            st.page_link("pages/analysis.py", label="Conversation Analysis", icon=":material/analytics:")
            st.page_link("pages/auto_eval.py", label="Automated Evaluation", icon=":material/quiz:")
            st.page_link("pages/users.py", label="User Management", icon=":material/group:")

        st.write("")

        if user := st.session_state.get("user_name"):
            with st.popover("⚙️&nbsp; Settings", use_container_width=True):
                st.write(f"Logged in user: `{user}`")
                sidebar_container = st.container()
                if st.button("🔑&nbsp; Logout", use_container_width=True):
                    st.session_state.user_name = None
                    st.session_state.admin_mode = None
                    if visibility != "public":
                        st.switch_page("app.py")
                    else:
                        st.rerun()
        else:
            sidebar_container = st.container()
            if st.button("🔑&nbsp; Login", use_container_width=True):
                login()

    return sidebar_container


@st.experimental_dialog("Login")
def login():
    conv_mgr: ConversationManager = st.session_state.conversation_manager
    options = set([""])
    options.update(conv_mgr.list_users())
    existing = st.selectbox("Existing user:", options)
    if not existing:
        new_user = st.text_input("New user:")
    admin_mode = st.checkbox("Admin mode", value=True)
    if st.button("Submit"):
        st.session_state.user_name = existing or new_user
        st.session_state.admin_mode = admin_mode
        st.rerun()


def configure_model(container, model_config: ModelConfig, key: str):
    MODEL_KEY = f"model_{key}"
    TEMPERATURE_KEY = f"temperature_{key}"
    TOP_P_KEY = f"top_p_{key}"
    MAX_NEW_TOKENS_KEY = f"max_new_tokens_{key}"

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = model_config.model
        st.session_state[TEMPERATURE_KEY] = model_config.temperature
        st.session_state[TOP_P_KEY] = model_config.top_p
        st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens

    with container:
        with st.popover(f"Configure :blue[{st.session_state[MODEL_KEY]}]", use_container_width=True):
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
