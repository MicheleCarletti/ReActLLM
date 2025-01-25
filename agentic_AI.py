"""
@author: Michele Carletti
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import huggingface_pipeline
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from sympy import sympify
from duckduckgo_search import DDGS as ddg

# Open source pre-trained LLM
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

llm_pipeline = (
    "text_generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

llm = huggingface_pipeline(pipeline=llm_pipeline)




