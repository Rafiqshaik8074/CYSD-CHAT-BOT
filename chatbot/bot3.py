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

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.retrievers import MultiVectorRetriever
from langchain.schema import Document
from collections import deque

from django.conf import settings
import os


# --- Initialize components ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# persist_directory = r"F:\Sahla\CYSD Chatbot\embeddings_chunks1"

persist_directory = os.path.join(
    settings.BASE_DIR,
    'chatbot',
    'embeddings_chunks1',
    'embeddings_chunks1'
)

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
llm = OllamaLLM(model="mistral:instruct", max_tokens=200)


# --- Retrieval Function ---

# Similarity Search
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
    return [doc.page_content for doc in results]


# --- Prompt memory storage ---
chat_history = deque(maxlen=4)  # Stores last 2 user–bot exchanges (2 × 2 = 4 turns)


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
    full_prompt = f"""You are a helpful assistant answering questions about CYSD (Centre for Youth and Social Development), co-founded by Jagadananda and Prafulla Kumar Sahoo in 1982.

- Speak clearly and with confidence — like someone who is part of CYSD.
- Do **not** refer to yourself (e.g., avoid phrases like "as an assistant" or "I can confidently say").
- Just answer the user's question as if you're directly explaining CYSD's work, values, and impact.
- Keep your answer concise and focused.
- Please don't ask the user to visit the website.

Chat History:
{memory_context}

CYSD Info:
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
