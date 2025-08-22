from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from rag.vector_store import get_vectorstore

# # Load LLAMA and QA system
# def load_llm():
#     return LlamaCpp(
#         #model_path=".\\models\\llama\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#         model_path=".\\models\\llama\\llama-2-7b.Q4_K_M.gguf",
#         n_ctx=2048,
#         n_threads=8,
#         n_gpu_layers=0,  # Optional, set 0 for CPU-only
#         temperature=0.7,
#         top_p=0.95,
#         max_tokens=256
#     )

# retriever = get_vectorstore().as_retriever()
# llm = load_llm()

# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

import json
import requests
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS # Example vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings # Example embeddings
from langchain.docstore.document import Document # For creating dummy documents

# --- Dummy Vectorstore and LLM Setup (for demonstration) ---
# In a real scenario, get_vectorstore() would load your actual fashion data.
# And load_llm() would load your LlamaCpp model.

# def generate_prompt(user_style):
#     response = qa_chain.run(f"Suggest a prompt for: {user_style}")
#     return response
def get_vectorstore():
    # This is a dummy vectorstore for demonstration purposes.
    # In a real application, you would load your actual vectorstore
    # with fashion-related documents.
    documents = [
        #Document(page_content="1920s flapper dresses were characterized by a loose, straight silhouette, dropped waists, and heavy embellishments like beads, sequins, and fringe. Popular fabrics included silk, chiffon, and velvet. Hair was often bobbed, and accessories included long pearl necklaces, cloche hats, and T-strap shoes."),
        Document(page_content="Modern streetwear often features oversized hoodies, cargo pants, sneakers, and bold graphic prints. Influences come from hip-hop culture, skateboarding, and urban environments. Materials like cotton, fleece, and nylon are common."),
        Document(page_content="Modern suits blend classic tailoring with contemporary style elements." \
                              " Common features include slim-fit or relaxed blazers, tapered trousers, and " \
                              "subtle or bold patterns. Influences range from traditional business attire to " \
                              "fashion-forward street style. Materials often include wool, linen, cotton blends, "
                              "and sometimes technical fabrics for flexibility and breathability. Accessories like pocket squares, " \
                              "minimalist watches, and loafers or sneakers add a personalized touch to the overall look."),
        #Document(page_content="Victorian gothic fashion incorporates elements like corsets, long skirts, lace, velvet, and dark colors, often inspired by mourning attire and romantic literature. Accessories include chokers, cameo jewelry, and elaborate hats.")
    ]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def load_llm():
    # This assumes you have the LlamaCpp library installed and the .gguf model file.
    # Replace with your actual model path and configuration.
    try:
        llm_instance = LlamaCpp(
            model_path=".\\models\\llama\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            #model_path=".\\models\\llama\\llama-2-7b.Q4_K_M.gguf",
             n_ctx=2048,
            n_threads=8,
            n_gpu_layers=0,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
            verbose=False,
        )
        return llm_instance
    except Exception as e:
        print(f"Error loading LlamaCpp model: {e}")
        print("Please ensure 'llama-cpp-python' is installed and the model path is correct.")
        print("You might need to download the .gguf model file and place it in the specified path.")
        # Return a mock LLM or raise an error to prevent further execution
        class MockLLM:
            def __call__(self, prompt):
                print(f"Mock LLM called with: {prompt[:100]}...")
                return "Mock LLM response: This is a placeholder response due to LLM loading error."
            def invoke(self, prompt):
                return self.__call__(prompt)
        return MockLLM()

# --- Agent Components ---
retriever = get_vectorstore().as_retriever()
llm = load_llm() # This LLM instance will be used for RAG and potentially for reasoning
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

async def generate_prompt(user_query: str) -> str:
    """
    An agentic approach to generating fashion-related image prompts.
    The agent decides whether to use RAG or refine the query based on its analysis.
    """
    print(f"\n--- Agent received query: '{user_query}' ---")

    # Step 1: Agent's Self-Reflection / Decision on Query Quality
    # Use the LLM to evaluate the user's initial query and suggest improvements
    analysis_prompt_template = f"""
    You are an AI assistant tasked with analyzing user queries for fashion image generation.
    Your goal is to determine if the query is sufficiently detailed and specific to generate a high-quality, visually rich image.
    If the query is vague or lacks descriptive elements, you should suggest how to improve it by adding details about era, style, fabric, mood, lighting, setting, or specific garment types.

    Respond in one of two formats:
    1. If the query is good enough: "DECISION: GOOD"
    2. If the query needs improvement: "DECISION: IMPROVE\nSUGGESTION: [Your detailed suggestion for improvement]"

    User Query: "{user_query}"
    """

    print("Agent is analyzing the user query for completeness...")
    # Make an LLM call for analysis. Use llm.invoke or llm.__call__ directly.
    # Note: LlamaCpp's invoke method is typically synchronous.
    analysis_response = llm.invoke(analysis_prompt_template)
    print(f"Agent's analysis response:\n{analysis_response}")

    needs_refinement = False
    refinement_suggestion = ""
    generated_prompt = ""

    if "DECISION: IMPROVE" in analysis_response:
        needs_refinement = True
        # Extract the suggestion part
        suggestion_start = analysis_response.find("SUGGESTION:")
        if suggestion_start != -1:
            refinement_suggestion = analysis_response[suggestion_start + len("SUGGESTION:"):].strip()
        print(f"Agent decided: Query needs refinement. Suggestion: '{refinement_suggestion}'")
    else:
        print("Agent decided: Query is sufficiently detailed.")

    final_prompt_query = user_query

    # Step 2: Agent's Action based on Decision (Refine or Use RAG)
    if needs_refinement and refinement_suggestion:
        # Agent decides to refine the prompt using the LLM and the suggestion
        refinement_instruction = f"""
        Given the original user query: "{user_query}"
        And the suggested improvements: "{refinement_suggestion}"
        Generate a new, more detailed and visually rich image generation prompt for a fashion image.
        Combine the original idea with the suggested details.
        """
        print("Agent is refining the prompt based on the suggestion...")
        # This would be another LLM call to generate the refined query
        final_prompt_query = llm.invoke(refinement_instruction)
        print(f"Agent refined query to: '{final_prompt_query}'")
    else:
        # If no refinement needed, or if suggestion was empty, use original query
        final_prompt_query = user_query
        print("Agent will use the original query for RAG.")

    # Step 3: Agent uses RAG for grounded generation of the final image prompt
    print("Agent is now consulting its fashion knowledge base (RAG) to generate the final image prompt.")
    # The qa_chain uses the LLM and retriever together
    generated_prompt = qa_chain.run(f"Based on the following, suggest a detailed image generation prompt for Stable Diffusion, incorporating fashion details: {final_prompt_query}")

    print("\n--- Agent Task Complete ---")
    print("Final Prompt Generated:", generated_prompt)
    return generated_prompt

# --- Example Usage ---
# To run these examples, you would typically use an async event loop if
# LlamaCpp calls were truly async. However, LlamaCpp's invoke is often synchronous.
# For a simple script, just call it directly.

# Example 1: Prompt needing improvement
# print("Running Example 1:")
# result1 = agentic_generate_fashion_prompt("a dress")
# print(f"\nResult 1: {result1}")

# Example 2: Prompt that is likely good enough
# print("\nRunning Example 2:")
# result2 = agentic_generate_fashion_prompt("A highly detailed 1920s flapper dress, with intricate beadwork and fringe, worn by a woman dancing in a smoky jazz club, cinematic lighting.")
# print(f"\nResult 2: {result2}")

# Example 3: Another prompt needing improvement
# print("\nRunning Example 3:")
# result3 = agentic_generate_fashion_prompt("winter clothes")
# print(f"\nResult 3: {result3}")
