# chatbot/rag.py
from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_ollama import OllamaEmbeddings

# Initialize components
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="llama3")
persist_dir = "chatbot/embeddings_store/embeddings_68"

# persist_dir = "chatbot\embeddings_store\embeddings_68\embeddings_68\8e93c5e9-1d4e-4fb1-b512-91390b1d5b86"
# persist_dir = "chatbot\embeddings_store\embeddings_68\embeddings_68"

vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
llm = Ollama(model="llama3")

# Retrieval + Response Generation
# def generate_response(query, top_k=3):
#     query_vector = embeddings.embed_query(query)
#     results = vectorstore.similarity_search_by_vector(query_vector, k=top_k)
#     context = "\n".join([doc.page_content for doc in results])

#     prompt = f"""You are a helpful assistant for CYSD.
# Use the following context to answer the query. Don't mention the word 'context'.

# Context:
# {context}

# Query: {query}

# Answer:"""

#     return llm.invoke(prompt).strip()



# -------------------------------------- Old Code ---------------------------------------

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
