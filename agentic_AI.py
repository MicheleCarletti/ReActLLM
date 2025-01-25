"""
@author: Michele Carletti
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from sympy import sympify
from duckduckgo_search import DDGS

# Open source pre-trained LLM
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    truncation=True  # Explicit truncation
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Tool for computation
def caluclator(expression: str) -> str:
    try:
        result = sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error while computing: {str(e)}"

# Search tool based on Duk Duck Go Engine
def web_search(query:str) -> str:
    try:
        results = DDGS().text(query, max_results=3)  # Gets first three results
        if not results:
            return "No result found"
        return " | ".join(result['title'] + ": " + result['href'] for result in results)
    except Exception as e:
        return f"Error while searching: {str(e)}"

# Toolbox
tools = [
    Tool(
        name="Calculator",
        func=caluclator,
        description="Use this tool to solve calculus"
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="Use this tool to retrieve web information through DuckDuckGo"
    )
]

# Prompt configuration for ReAct agent
react_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=(
        "You're a smart agent able to reason and act for problem solving tasks.\n"
        "Follow this structure:\n"
        "1. Reason about the problem.\n"
        "2. Identify a proper action.\n"
        "3. Execute the action and update you're reason.\n\n"
        "Example:\n"
        "Question: What is the capital of France?\n"
        "Thought: The question asks the capita of France. I can found this information.\n"
        "Action: Use WebSearch tool to find the answer.\n"
        "Result: The capital of france is Paris.\n\n"
        "Now answer the following question.\n\n"
        "Queistion: {input}\n"
        "{agent_scratchpad}"

    ),
)

# Agent configuration
react_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Launch the agent
if __name__ == "__main__":
    print("ReAct Agent running...")
    while True:
        user_input = input("Ask something (quit to stop): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bye ...")
            break
        response = react_agent.invoke(user_input)
        print(f"Agent: {response}")








