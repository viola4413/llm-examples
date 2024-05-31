from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.litellm import LiteLLM
import streamlit as st
import numpy as np
import os

# replicate key for running feedback
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

@st.cache_resource
def create_provider():
    return LiteLLM(model_engine="replicate/snowflake/snowflake-arctic-instruct")

@st.cache_resource
def create_base_feedback_fns():
    # set feedback functions for trulens to use
    provider = create_provider()
    f_criminality_input = Feedback(provider.criminality, name = "Criminality input", higher_is_better=False).on(Select.RecordInput)
    f_criminality_output = Feedback(provider.criminality, name = "Criminality output", higher_is_better=False).on(Select.Record.app.retrieve_and_generate_stream.rets[:].collect())
    feedbacks = [f_criminality_input, f_criminality_output]
    return feedbacks

@st.cache_resource
def create_context_relevance_feedback_fns():
    provider = create_provider()
    f_context_relevance = (
                Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
                .on_input()
                .on(Select.RecordCalls.retrieve_context.rets[1][:].node.text)
                .aggregate(np.mean) # choose a different aggregation method if you wish
            )
    return f_context_relevance
