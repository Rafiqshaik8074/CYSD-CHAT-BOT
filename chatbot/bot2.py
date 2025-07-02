# Code for deployment

'''
Installations:
    
Downlod Ollama for windows
    
# Open anaconda prompt / command prompt
pip install langchain
ollama pull llama3
#pip install chromadb
'''
# Import Libraries

# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import OllamaLLM

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.retrievers import MultiVectorRetriever
from langchain.schema import Document
from collections import deque

from django.conf import settings
import os

# --- Initialize components ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# persist_directory = r"F:\Sahla\CYSD Chatbot\embeddings_cysd"


persist_directory = os.path.join(
    settings.BASE_DIR,
    'chatbot',
    'embeddings_cysd'
)


vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
llm = OllamaLLM(model="llama3", max_tokens=500)


# --- Retrieval Function ---
def retrieve_relevant_chunks(query, top_k=5):
    query_vector = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
    return [doc.page_content for doc in results]

# --- Prompt memory storage ---
chat_history = deque(maxlen=6)  # Stores last 3 user–bot exchanges (2 × 3 = 6 turns)

# --- LLM Response Generation with Memory ---
def generate_response(query):
    # Add user query to history
    chat_history.append(f"User: {query}")

    # Retrieve chunks
    retrieved_texts = retrieve_relevant_chunks(query)
    context = "\n".join(retrieved_texts)

    # Format chat memory
    memory_context = "\n".join(chat_history)

    # Prepare prompt
    '''
    #full_prompt = f"""You are an AI chatbot answering questions about CYSD (Centre for Youth and Social Development), a nonprofit based in Odisha, India.
    full_prompt = f"""You are an expert assistant representing CYSD (Centre for Youth and Social Development), providing clear, confident answers about its work, mission, and impact in Odisha, India. Respond with authority and avoid phrases like ‘according to the website’ or ‘based on the context.’ Speak as if you know, because you do.
Avoid saying "based on the context" or similar phrases.

Chat History:\n{memory_context}

Website Context:\n{context}

User Query: {query}
Answer:"""
'''
    full_prompt = f"""You are a helpful assistant answering questions about CYSD (Centre for Youth and Social Development), a nonprofit based in Odisha, India.

- Speak clearly and with confidence — like someone who is part of CYSD.
- Do **not** refer to yourself (e.g., avoid phrases like "as an assistant" or "I can confidently say").
- Do **not** mention "the website" or "the retrieved context."
- Just answer the user's question as if you're directly explaining CYSD's work, values, and impact.

Chat History:
{memory_context}

Website Context:
{context}

User Query: {query}
Answer:"""

    response = llm.invoke(full_prompt).strip()
    
    # Store bot response in memory
    chat_history.append(f"Bot: {response}")
    return response

# --- Continuous Chat Loop ---
# def chat_loop():
#     print("Chatbot: Hello! Ask me anything about CYSD. Type 'exit' to end.")
#     while True:
#         user_query = input("\nYou: ")
#         if user_query.lower() == "exit":
#             print("\nChatbot: Goodbye!")
#             break
#         response = generate_response(user_query)
#         print(f"Chatbot: {response}")

# chat_loop()
