import json
import pathlib
import threading
from typing import Dict

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from conversation_manager import ConversationManager
from llm import StreamGenerator, AVAILABLE_MODELS
from schema import (
    Conversation,
    Message,
    ModelConfig,
)

from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.litellm import LiteLLM

import numpy as np

# feedback functions
from trulens_eval import TruCustomApp

generator = StreamGenerator()

def page_setup(title, wide_mode=False, collapse_sidebar=False, visibility="public"):
    if st.get_option("client.showSidebarNavigation") and "already_ran" not in st.session_state:
        st.set_option("client.showSidebarNavigation", False)
        st.session_state.already_ran = True
        st.session_state.use_rag = True
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

@st.cache_resource
def create_feedback_fns():
    # set feedback functions for trulens to use
    provider = LiteLLM(model_engine="replicate/snowflake/snowflake-arctic-instruct")
    f_context_relevance = (
            Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
            .on_input()
            .on(Select.RecordCalls.retrieve_context.rets[1][:].node.text)
            .aggregate(np.mean) # choose a different aggregation method if you wish
        )
    f_criminality_input = Feedback(provider.criminality, name = "Criminality input", higher_is_better=False).on(Select.RecordInput)
    f_criminality_output = Feedback(provider.criminality, name = "Criminality output", higher_is_better=False).on(Select.Record.app.retrieve_and_generate_stream.rets[:].collect())
    return [f_context_relevance, f_criminality_input, f_criminality_output]

@st.cache_resource
def get_tru_app_id(model: str, temperature: float, top_p: float, max_new_tokens: int, use_rag: bool):
    # Args used for caching
    if 'app_id_iterator' not in st.session_state:
        st.session_state['app_id_iterator'] = 0
    app_idx = st.session_state.get('app_id_iterator')
    app_id = f"App {app_idx}"
    st.session_state['app_id_iterator'] += 1
    return app_id

def configure_model(*, container, model_config: ModelConfig, key: str, full_width: bool = True):
    MODEL_KEY = f"model_{key}"
    TEMPERATURE_KEY = f"temperature_{key}"
    TOP_P_KEY = f"top_p_{key}"
    MAX_NEW_TOKENS_KEY = f"max_new_tokens_{key}"
    SYSTEM_PROMPT_KEY = f"system_prompt_{key}"
    USE_RAG_KEY = "use_rag"

    # initialize app metadata for tracking
    metadata = {
        "model": st.session_state.get(MODEL_KEY, model_config.model),
        "temperature": st.session_state.get(TEMPERATURE_KEY, model_config.temperature),
        "top_p": st.session_state.get(TOP_P_KEY, model_config.top_p),
        "max_new_tokens": st.session_state.get(MAX_NEW_TOKENS_KEY, model_config.max_new_tokens),
        "use_rag": st.session_state.get(USE_RAG_KEY, model_config.use_rag),
    }

    if MODEL_KEY not in st.session_state:
        st.session_state[MODEL_KEY] = model_config.model
        st.session_state[TEMPERATURE_KEY] = model_config.temperature
        st.session_state[TOP_P_KEY] = model_config.top_p
        st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens
        st.session_state[USE_RAG_KEY] = model_config.use_rag
        metadata = {
                        "model": st.session_state[MODEL_KEY],
                        "temperature": st.session_state[TEMPERATURE_KEY],
                        "top_p": st.session_state[TOP_P_KEY],
                        "max_new_tokens": st.session_state[MAX_NEW_TOKENS_KEY],
                        "use_rag": st.session_state[USE_RAG_KEY],
                    }

    with container:
        with st.popover(
            f"Configure :blue[{st.session_state[MODEL_KEY]}]", use_container_width=full_width
        ):
            left1, right1 = st.columns(2)
            left2, right2 = st.columns(2)
            with left1:
                model_config.model = st.selectbox(
                    label="Select model:",
                    options=AVAILABLE_MODELS,
                    key=MODEL_KEY,
                )
                if model_config.model != st.session_state[MODEL_KEY]:
                    st.session_state[MODEL_KEY] = model_config.model

            with left2:
                SYSTEM_PROMPT_HELP = """
                    Add a system prompt which is added to the beginning
                    of each conversation.
                """
                model_config.system_prompt = st.text_area(
                    label="System Prompt:",
                    height=2,
                    key=SYSTEM_PROMPT_KEY,
                    help=SYSTEM_PROMPT_HELP,
                )
                if model_config.system_prompt != st.session_state[SYSTEM_PROMPT_KEY]:
                    st.session_state[SYSTEM_PROMPT_KEY] = model_config.system_prompt

            with right1:
                model_config.temperature = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    label="Temperature:",
                    key=TEMPERATURE_KEY,
                )
                if model_config.temperature != st.session_state[TEMPERATURE_KEY]:
                    st.session_state[TEMPERATURE_KEY] = model_config.temperature

            with right2:
                model_config.top_p = st.slider(
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                    label="Top P:",
                    key=TOP_P_KEY,
                )
                if model_config.top_p != st.session_state[TOP_P_KEY]:
                    st.session_state[TOP_P_KEY] = model_config.top_p

                model_config.max_new_tokens = st.slider(
                    min_value=100,
                    max_value=1500,
                    step=100,
                    label="Max new tokens:",
                    key=MAX_NEW_TOKENS_KEY,
                )
                if model_config.max_new_tokens != st.session_state[MAX_NEW_TOKENS_KEY]:
                    st.session_state[MAX_NEW_TOKENS_KEY] = model_config.max_new_tokens
                
                model_config.use_rag = st.toggle(
                    label="Access to Streamlit Docs",
                    value=True,
                    key=USE_RAG_KEY
                )
                if model_config.use_rag != st.session_state.use_rag:
                    st.session_state.use_rag = model_config.use_rag

    app_id = get_tru_app_id(**metadata)
    feedbacks = create_feedback_fns()
    app = TruCustomApp(generator, app_id=app_id, metadata=metadata, feedbacks=feedbacks)
    st.session_state['trulens_recorder'] = app
    print(model_config)
    return model_config

def chat_response(
    conversation: Conversation,
    container=None,
):
    conversation.add_message(
        Message(role="assistant", content=""),
        render=False,
    )
    try:
        if st.session_state['use_rag']:
            user_message, prompt = generator.prepare_prompt(conversation)
            stream_iter = generator.retrieve_and_generate_stream(user_message, prompt, conversation) # hack - not displaying in dashboard without this
            with st.session_state['trulens_recorder']:
                user_message, prompt = generator.prepare_prompt(conversation)
                generator.retrieve_and_generate_stream(user_message, prompt, conversation)
        else:
            #user_message, prompt = generator.prepare_prompt(conversation)
            #stream_iter = generator.generate_stream(user_message, prompt, conversation) # hack - not displaying in dashboard without this
            with st.session_state['trulens_recorder']:
                user_message, prompt = generator.prepare_prompt(conversation)
                generator.generate_stream(user_message, prompt, conversation)

        def generate_and_save():
            for t in stream_iter:
                conversation.messages[-1].content += str(t)
                yield str(t)

        if container:
            chat = container.chat_message("assistant")
        else:
            chat = st.chat_message("assistant")
        full_streamed_response = chat.write_stream(generate_and_save)
        conversation.messages[-1].content = str(full_streamed_response).strip()
    except Exception as e:
        conversation.has_error = True
        print("Error while generating chat response: " + str(e))


def generate_title(
    user_input: str,
    response_dict: Dict,
):
    SYSTEM_PROMPT = """
        You are a helpful assistant generating a brief summary title of a
        conversation based on the users input. The summary title should
        be no more than 4-5 words, with 2-3 words as a typical response.
        In general, brief is better when the title is a clear summary.

        Input will be provided in JSON format and you should specify the
        output in JSON format. Do not add any commentary or discussion.
        ONLY return the JSON.

        Here are a few examples:
        INPUT: {"input": "Hey, I'm looking for tips on planning a trip to Chicago. What should I do while I'm there?"}
        OUTPUT: {"summary": "Visiting Chicago"}

        INPUT: {"input": "I've been scripting and doing simple database work for a few years and I want to learn frontend web development. Where should I start?"}
        OUTPUT: {"summary": "Learning frontend development"}

        INPUT: {"input": "Can you share a few jokes?"}
        OUTPUT: {"summary": "Sharing jokes"}

        Ok, now your turn. Remember to only respond with the JSON.
        ------------------------------------------
    """
    conversation = Conversation()
    conversation.model_config = ModelConfig(system_prompt=SYSTEM_PROMPT)
    input_msg = json.dumps({"input": user_input})
    conversation.add_message(Message(role="user", content=input_msg), render=False)
    title_json = ""
    try:
        last_user_message, prompt = generator.prepare_prompt(conversation)
        stream_iter = generator.generate_stream(last_user_message, prompt, conversation)
        for t in stream_iter:
            title_json += str(t)
        result = json.loads(title_json)
        response_dict["output"] = result["summary"]

    except Exception as e:
        response_dict["error"] = True
        print("Error while generating title: " + str(e))
        print("Response:" + title_json)


def st_thread(target, args) -> threading.Thread:
    """Return a function as a Streamlit-safe thread"""

    thread = threading.Thread(target=target, args=args)
    add_script_run_ctx(thread, get_script_run_ctx())
    return thread

