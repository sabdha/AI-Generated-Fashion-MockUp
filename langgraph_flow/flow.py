from langgraph.graph import StateGraph
from rag.prompt_engine import generate_prompt
from inference.generate import generate_image
from typing import TypedDict
import asyncio
class StylistState(TypedDict):
    query: str
    answer: str
    image_path: str

def mock_llm(state: StylistState) -> StylistState:
    return {
        "query": state["query"],
        "answer": f"Mock answer to: {state['query']}"
    }

# Define the pipeline flow
def build_flow():
    builder = StateGraph(StylistState) 

    # Step 1: Generate prompt from user style using RAG + LLAMA
    def prompt_node(state):
        user_style = state["query"]
        prompt = asyncio.run(generate_prompt(user_style))
        return {"answer": prompt}

    # Step 2: Generate image using LoRA
    def image_node(state):
        prompt = state["answer"]
        output_path = generate_image(prompt)
        return {"image_path": output_path}

    builder.add_node("GeneratePrompt", prompt_node)
    builder.add_node("GenerateImage", image_node)

    builder.set_entry_point("GeneratePrompt")
    builder.add_edge("GeneratePrompt", "GenerateImage")
    builder.set_finish_point("GenerateImage")

    return builder.compile()
