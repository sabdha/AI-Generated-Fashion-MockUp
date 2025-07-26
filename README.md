# Agent-Driven Fashion Mock-Up with LoRA-Enhanced Diffusion and Query Optimization
<pre>
Designed a self-reflective LLM agent that iteratively analyzes user queries to determine ambiguity. Based on confidence scores and heuristics, the agent either rewrites the query for clarity or performs retrieval from a fashion-style vector database. This intelligent routing ensures high-quality prompt generation for Stable Diffusion-based virtual try-on image synthesis.
</pre>  
# Agent-Driven Product Mock-Up with LoRA-Enhanced Diffusion and Query Optimization for Fashion/Retail Campaigns   
## Sample video and images:  
https://github.com/user-attachments/assets/06fa3180-1806-472d-97f8-74be9585d294  

<img width="413" height="392" alt="image" src="https://github.com/user-attachments/assets/b4c07e75-9a02-48a7-963f-a395e6d84960" />
<img width="421" height="320" alt="image" src="https://github.com/user-attachments/assets/688cecd0-c1cc-425c-b856-bc75aa5ce8ed" />

<pre>
Problem  
Marketing and design teams often require high-quality, style-consistent mock-ups of fashion products (e.g., dresses, shirts,
accessories) in various poses, styles, and backgrounds — well before physical prototypes are available. Manual design iterations
are time-consuming and inconsistent.
  
Solution
A proof-of-concept system that generates fashion mockups by combining fast generative models with intelligent language processing:

  User Input: Users upload a few reference photos or provide a style/text prompt (e.g., “A red party dress in a studio background”).
  LLM Agent Decision Layer: 
    A self-reflective agent (LLAMA) analyzes the query through 3 stages:
    -  Checks prompt clarity.
    -  Decides whether to rewrite or retrieve from a vector database (RAG).
    -  Forwards the final refined input to the generation pipeline.
  Generation & Flow Management
  -  LangGraph manages the end-to-end pipeline: input → decision → image generation → text generation → output.
  -  LoRA-tuned Stable Diffusion models generate mock-up images aligned with the user’s style or pose.
  -  RAG + LLAMA generates stylized captions or product marketing copy.  
</pre>
# System Architecture  
<pre>
+------------------+
|   User Query     |
+------------------+
         ↓
+---------------------------+
|    LLM Agent (Multi-Pass) |  ← decides:
|   - Query Analyzer        |      • Is input vague?
|   - Rewrite or RAG?       |      • Should we guide or retrieve?
+---------------------------+
    ↓                     ↓
Prompt Rewrite     RAG (FashionVectorDB)
    ↓                     ↓
       +-------------------------------+
       | Stable Diffusion + LoRA/SDXL |
       +-------------------------------+
                      ↓
       Generated Fashion Mockup Image
  
</pre>

# 🔧 Tech Stack   
________________________________________  
📦 1. Model Inference & Generation
<pre>
[🤗 diffusers]	           --> Stable Diffusion pipeline (runwayml/stable-diffusion-v1-5)  
LoRA (Low-Rank Adaptation) -->	Inject fine-tuned weights into "to_q" and "to_v" attention modules  
safetensors                -->	Load LoRA weights safely and efficiently  
PyTorch                    -->	Backbone for all ML models (LLM and image generation)  
Custom LoRA Loader         -->	manually merged LoRA weights  
</pre>
________________________________________    
🌐 2. LLM, RAG & Orchestration  
<pre>   
[LLAMA (LLaMA2)] -->	Local LLM for prompt understanding and command generation  
[LangChain]      -->	RAG (Retrieval-Augmented Generation) framework connecting LLM to knowledge base  
[LangGraph]      -->	Graph-based orchestration of agents/tasks  
[Vector DB]      -->  (Chroma)	Stores embedded documents for RAG  
[RAG Pipeline]   -->	Search + retrieval + context-injection into LLAMA prompt  
Embedding Model  -->	Hugging Face embeddings  
</pre>
________________________________________  
🖥️ 3. Frontend (UI Layer)  
<pre>    
[Streamlit]       -->	Interactive UI: text prompt input, image generation  
OpenCV / PIL      -->	Display and manipulate generated images  
Streamlit Widgets -->	Input: st.text_input, st.button / Output: st.image, etc.  
</pre>
________________________________________  
🧰 4. Development Tools & Environment   
<pre>  
Python 3.11+ --> Core language  
Torch        -->	Core ML library  
virtualenv   -->	Dependency isolation  
VSCode       -->	Development IDEs  
</pre>

