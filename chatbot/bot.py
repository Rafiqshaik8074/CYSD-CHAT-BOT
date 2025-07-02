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
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


from django.conf import settings
import os




# --- Initialize components ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# persist_directory = r"C:\Users\nuha_faizal\Desktop\CYSD Chatbot\embeddings_68"
# persist_directory = "chatbot/embeddings_store/embeddings_68"
# persist_directory = r"chatbot\embeddings_store\embeddings_68"
# persist_directory = r"C:\Users\CHIST\Desktop\CHAT-BOT\cysd_chatbot\chatbot\embeddings_store\embeddings_68\embeddings_68"


persist_directory = os.path.join(
    settings.BASE_DIR,
    'chatbot',
    'embeddings_68'
)

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
llm = OllamaLLM(model="llama3")


# --- Retrieval Function ---
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
    return [doc.page_content for doc in results]

# --- LLM Response Generation ---
def generate_response(query):
    retrieved_texts = retrieve_relevant_chunks(query)
    context = "\n".join(retrieved_texts)
    
    full_prompt = full_prompt = f"""You are an AI chatbot answering queries based on CYSD's website.
    Please don't use phrases like "according to the website" or "based on the context."\n
Context:\n{context}\n
Query: {query}
Answer:"""

    response = llm.invoke(full_prompt)
    return response.strip()

# Continuous Chat using While Loop

# This allows continuous interaction with the chatbot.

def chat_loop():
    print("Chatbot: Hello! Ask me anything about CYSD. Type 'exit' to end.")

    while True:
        user_query = input("\nYou: ")

        if user_query.lower() == "exit":
            print("\nChatbot: Goodbye!")
            break

        response = generate_response(user_query)
        print(f"Chatbot: {response}")

# chat_loop()


