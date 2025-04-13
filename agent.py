from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv("api.env")

####Gemini_model#####
Gemini_api_key = os.getenv("gemini_api_key")
rag_llm = ChatGoogleGenerativeAI(
    model = "gemini-2.0-flash",
    api_key = Gemini_api_key
)
#######################

class RetrieValEvaluator(BaseModel):
    binary_score : str = Field(
        description="Document are relevant to the question, 'yes' or 'no' "
    )
structure_llm_evaluator = rag_llm.with_structured_output(RetrieValEvaluator)
system = """You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question. \\n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \\n
    Output a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
retrieve_output_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", system),
    ("human", "Retrieved_document : \\n\n {document} User question : {question}"),
    ]
)
retrieve_grader = retrieve_output_prompt | structure_llm_evaluator
#############################

systems = """You are a question re-writer that converts an input question to a better version that is optimized \\n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_writer_promp = ChatPromptTemplate.from_messages(
    [
        ("system", systems),
        (
            "human",
            "here is the initial question: \\n\n {question}\\n Formulated an improve question"
        ),
    ]
)
question_rewriter = retrieve_output_prompt | rag_llm | StrOutputParser()