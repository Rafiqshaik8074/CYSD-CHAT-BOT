# Create a Chatbot for CYSD Website with Hybrid Approach (Rule and RAG based)

#---

# Rule Based Approach

# pip install textblob
# pip install fuzzywuzzy
# pip install python-Levenshtein

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from fuzzywuzzy import process  # Fuzzy string matching
from fuzzywuzzy import fuzz
from functools import lru_cache

# Download necessary resources
import nltk
from nltk.data import find

def download_nltk_resource(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Predefined responses
responses = {

    # About CYSD
    "cysd": "CYSD (Centre for Youth and Social Development) is a non-governmental organization focused on improving the quality of life for marginalized communities in Odisha, India.",
    "cysd stand": "CYSD stands for the Centre for Youth and Social Development.",
    "cysd full form": "Full form of CYSD is Centre for Youth and Social Development.",
    "founded": "CYSD was founded in 1982.",
    "location": "CYSD is based in Bhubaneswar, Odisha, India.",
    "co-founders": "Jagadananda and Prafulla Kumar Sahoo.",
    "chairperson": "Dr. Rajesh Tandon is the current Chairperson of CYSD.",

    # CYSD’s Work & Focus Areas
    "focus areas": "CYSD works in three key areas: Sustainable Livelihoods, Participatory Governance, and Climate Change & Disaster Risk Reduction.",
    
    # Contact & Location
    "contact": "You can reach us at cysd@cysd.org or call (0674) 2300983 / 2301725.",
    "email": "CYSD’s official email is cysd@cysd.org.",
    "phone": "You can call CYSD at (0674) 2300983 or (0674) 2301725.",
    "address": "CYSD is located at E-1, Institutional Area, Near Xavier Square, Bhubaneswar-751013, Odisha, India.",
    "visit office": "You can visit CYSD’s office at E-1, Institutional Area, Near Xavier Square, Bhubaneswar-751013, Odisha, India.",

    # Volunteering, Careers, and Internships
    "volunteer": "To volunteer, visit our website or contact us at cysd@cysd.org.",
    "job": "Job openings at CYSD are updated on our Careers page. Please check our website for current vacancies.",

    # Donations & Partnerships
    "donate": "You can donate online through our website’s 'Donate' section.",
    "use donations": "CYSD ensures that donations directly support community development projects. Transparency reports are available on our website.",
    "business partnership": "Yes, CYSD collaborates with businesses and organizations to drive social impact.",
    "collaborate": "If you're interested in collaborating, contact us at cysd@cysd.org.",

}

 
# Synonym mapping for CYSD-related queries
synonym_map = {
    "full form": "cysd stand",
    "established": "founded",
    "founder": "co-founders",
    "cost": "donate",
    "rate": "donate",
    "charge": "donate",
    "help": "volunteer",
    "support": "volunteer",
    "assistance": "volunteer",
    "participate": "volunteer",
    "register": "volunteer",
    "sign up": "volunteer",
    "fund": "donate",
    "contribute": "donate",
    "sponsor": "donate",
    "career": "job",
    "vacancy": "job",
    "opening": "job",
    "position": "job",
    "number": "contact",
    "phone": "contact",
    "email": "contact",
    "reach out": "contact",
    "work": "focus areas",
    "services": "focus areas",
    "service": "focus areas",
    "student": "internship",
    "chairman": "chairperson",       
}


# Function to get synonyms
def get_synonym(word):
    return synonym_map.get(word.lower(), word)

# Function to preprocess user input
def preprocess_input(user_input):
    user_input = user_input.lower()

    # Spell correction
    user_input = str(TextBlob(user_input).correct())

    # Tokenization
    words = word_tokenize(user_input)

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stopwords
    filtered_words = [word for word in lemmatized_words if word not in stopwords.words('english')]
    
    # Apply synonym mapping
    processed_words = [get_synonym(word) for word in filtered_words]

    return " ".join(processed_words)

# Function to find the best response using fuzzy matching
def get_best_response(user_input):
    best_match, score = process.extractOne(user_input, responses.keys(), scorer=fuzz.token_sort_ratio)  # Find closest match

    if score > 60:  # If similarity is above threshold
        return responses[best_match]
    else:
        return "I'm sorry, I don't understand. Can you rephrase?"

# RAG Based Approach

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


# --- Initialize components ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = r"F:\Sahla\CYSD Chatbot\embeddings_chunks1"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
llm = OllamaLLM(model="mistral:instruct", max_tokens=100)


# --- Retrieval Function ---

# MMR (Maximal Marginal Relevance) Retrieval
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 30, "lambda_mult": 0.5}
)

#def retrieve_relevant_chunks(query, top_k=3):
def retrieve_relevant_chunks(query):
    results = retriever.invoke(query)
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


# --- Hybrid Approach ---

def hybrid_chatbot(user_input):
    # Step 1: Preprocess and check rule-based responses
    processed_input = preprocess_input(user_input)
    rule_response = get_best_response(processed_input)

    # Step 2: Use rule-based response if found
    if rule_response and "I don't understand" not in rule_response:
        return rule_response

    # Step 3: Fall back to RAG-based generation
    return generate_response(user_input)


# --- Continuous Chat Loop ---

def chat_loop():
    print("Chatbot: Hello! Ask me anything about CYSD. Type 'exit' to end.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == "exit":
            print("\nChatbot: Goodbye!")
            break

        response = hybrid_chatbot(user_query)
        print(f"Chatbot: {response}")

chat_loop()

#------------------------------------------------------------------------------

