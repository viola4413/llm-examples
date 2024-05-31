from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.hugs import Huggingface
from trulens_eval.feedback.provider.litellm import LiteLLM
import streamlit as st
import numpy as np
import os

# replicate key for running feedback
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

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
