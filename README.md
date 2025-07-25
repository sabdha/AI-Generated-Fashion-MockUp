# AI-Generated-Fashion-MockUp
<pre>
AI-Powered Product Mockups + Caption Generator using LoRA + Stable Diffusion + LLAMA + LangGraph + RAG
</pre>  
# AI-Generated Product Mock-ups for Fashion/Retail Campaigns   
## Sample video and images:  
https://github.com/user-attachments/assets/06fa3180-1806-472d-97f8-74be9585d294  

<img width="413" height="392" alt="image" src="https://github.com/user-attachments/assets/b4c07e75-9a02-48a7-963f-a395e6d84960" />
<img width="421" height="320" alt="image" src="https://github.com/user-attachments/assets/688cecd0-c1cc-425c-b856-bc75aa5ce8ed" />

<pre>
Problem:
  
Marketing teams often need high-quality mock-ups of clothing (e.g., dresses, shirts, accessories) in different styles, poses,
backgrounds — before the physical product exists.
  
Solution:
  
A PoC where a user:
  1.	Uploads a few reference photos or descriptions.
  2.	Chooses a prompt (e.g., “A red dress for party”).
  3.	The system:
        o	Uses LoRA-tuned Stable Diffusion to generate styled product images.
        o	Uses LLAMA + RAG to help write ad copy or suggest styling tips.
        o	Uses Lang Graph to manage flow (input → generation → captioning → approval/export).  
</pre>
# System Architecture  
<pre>
User Query  
    ↓  
LLAMA + RAG  
    ↓  
LangGraph (detects try-on intent)  
    ↓  
FashionVectorDB (gets clothing style vector/image)  
    ↓  
Workflow Decision AI (builds input for image generation)  
    ↓  
Stable Diffusion + LoRA (Style + Person)  
    ↓  
AI-Generated Image`  
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

