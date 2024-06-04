from trulens_eval import Feedback, Select
from trulens_eval.feedback.provider.litellm import LiteLLM
import numpy as np


provider = LiteLLM(model_engine="replicate/snowflake/snowflake-arctic-instruct")

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(Select.RecordCalls.retrieve_context.rets[1][:].node.text)
    .aggregate(np.mean) # choose a different aggregation method if you wish
)
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on_input()
    .on(Select.Record.app.retrieve_and_generate_response.rets[0])
    .aggregate(np.mean)
)
f_criminality_input = (
    Feedback(provider.criminality_with_cot_reasons,
             name = "Criminality input",
             higher_is_better=False)
             .on(Select.RecordInput)
)
f_criminality_output = (
    Feedback(provider.criminality_with_cot_reasons,
             name = "Criminality output",
             higher_is_better=False)
             .on(Select.Record.app.retrieve_and_generate_response.rets[0])
)

feedbacks_rag = [f_context_relevance, f_answer_relevance, f_criminality_input, f_criminality_output]
feedbacks_no_rag = [f_answer_relevance, f_criminality_input, f_criminality_output]
