import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from llm import generate_stream
from schema import Conversation, Message, ModelConfig


AVAILABLE_MODELS = [
    "snowflake/snowflake-arctic-instruct",
    "meta/meta-llama-3-8b",
    "mistralai/mistral-7b-instruct-v0.2",
]


def page_setup(title, icon, wide_mode=False, collapse_sidebar=False, public=True):
    if "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.rerun()

    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide" if wide_mode else "centered",
        initial_sidebar_state="collapsed" if collapse_sidebar else "auto",
    )

    st.title(f"{icon} {title}")

    # Add page navigation
    with st.sidebar:
        st.image("tasty_bytes_banner.png")
        st.title("Tasty Bytes LLM Eval")
        st.caption(
            "[Intro to Tasty Bytes](https://quickstarts.snowflake.com/guide/tasty_bytes_introduction/)"
        )

        st.write("")

        st.page_link("app.py", label="Direct Chat", icon="💬")
        st.page_link("pages/about.py", label="About", icon="ℹ️")

        if st.session_state.get("user_name"):
            st.page_link("pages/personal.py", label="Personal Stats", icon="🧑‍🚀")

        if st.session_state.get("admin_mode"):
            st.subheader("Admin view")
            st.page_link("pages/analysis.py", label="Conversation Analysis", icon="🔬")
            st.page_link("pages/auto_eval.py", label="Automated Evaluation", icon="🤖")
            st.page_link("pages/users.py", label="User Management", icon="👥")

        st.write("")
        st.write("")

        if not st.session_state.get("user_name"):
            if st.button("Login", use_container_width=True):
                login()
        else:
            st.write(f"Logged in user: `{st.session_state.user_name}`")
            if st.button("Logout", use_container_width=True):
                st.session_state.user_name = None
                st.session_state.admin_mode = None
                if not public:
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


def configure_model(container, model_id):
    MODEL_KEY = f"model_{model_id}"
    if MODEL_KEY in st.session_state:
        model_label = st.session_state.get(MODEL_KEY)
    else:
        model_label = AVAILABLE_MODELS[0]

    with container:
        with st.popover(f"Configure {model_id}: `{model_label}`", use_container_width=True):
            model = st.selectbox(
                label="Select model:", options=AVAILABLE_MODELS, key=f"model_{model_id}"
            )

            temperature = st.slider(
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                label="Temperature:",
                key=f"temp_{model_id}",
            )

            top_p = st.slider(
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                label="Top P:",
                key=f"top_p_{model_id}",
            )

            max_new_tokens = st.slider(
                min_value=100,
                max_value=1500,
                value=1024,
                step=100,
                label="Max new tokens:",
                key=f"max_tokens_{model_id}",
            )
    return ModelConfig(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def chat_response(
    conversation: Conversation,
    model_config: ModelConfig,
    container=None,
):
    stream_iter = generate_stream(
        conversation,
        model_config,
    )

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
