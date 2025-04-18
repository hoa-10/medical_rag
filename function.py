from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

import os
from agent import retrieve_grader, question_rewriter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_tavily import TavilySearch
from conversationMemory import ConservationMemory
from Retriever_memory import SemanticMemoryRetriever
from datetime import datetime
from translation_query_language import translate_vi2en
from processing_document import process_pdf_documents, query_document
# Initialize environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBWZUIJAUu4PYGXH7lSfUS9mjUgTdK7CWc")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")



rag_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY
)

web_search_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic='general'
)

# Initialize memory systems
memory = ConservationMemory()
semantic_memory = SemanticMemoryRetriever()
semantic_memory.add_memory(memory.get_all_history())

class State(TypedDict):
    question: str
    generation: str
    web_search: str
    document: List[Document]
    chat_history: List[Dict[str, str]]
    memory_context: str


def _get_query_text(question: Any) -> str:
    """Extract query text from question object or string."""
    if isinstance(question, dict):
        return question.get("text", str(question))
    return str(translate_vi2en(question))


def retrieve_memory(state: State) -> Dict[str, Any]:
    """Retrieve relevant memories based on the question."""
    print("---RETRIEVING MEMORIES---")
    
    query = _get_query_text(state["question"])
    memory_docs = semantic_memory.retrieve_relevant_memories(query, top_k=3)
    
    if memory_docs:
        print(f"Retrieved {len(memory_docs)} relevant memories")
        memory_context = "\n\n".join([doc.page_content for doc in memory_docs])
    else:
        print("No relevant memories found")
        memory_context = "No relevant past conversations found."
    
    return {
        "question": state["question"],
        "memory_context": memory_context
    }


def retrieve(state: State) -> Dict[str, Any]:
    """Retrieve documents based on the question using available retrievers."""
    print("--RETRIEVE--")
    
    query = _get_query_text(state["question"])
    print(f"Query: {query}")
    
    documents = []

    try:
        # Sử dụng hàm query_document thay vì trực tiếp gọi processing_pdf_document
        # Điều này sẽ tái sử dụng vector store nếu đã tồn tại
        retriever = process_pdf_documents(pdf_dir='pdf_database')
        
        if retriever:
            main_docs = query_document(query, retriever)
            documents.extend(main_docs)
            print(f"Retrieved documents from main retriever: {len(main_docs)}")
        else:

            main_docs = query_document(query)
            if main_docs:
                documents.extend(main_docs)
                print(f"Retrieved documents using direct query: {len(main_docs)}")
            else:
                print("Không thể truy vấn từ vector database")
    except Exception as e:
        print(f"Error retrieving from vector database: {e}")
        # Thử truy vấn trực tiếp nếu có lỗi
        try:
            fallback_docs = query_document(query)
            if fallback_docs:
                documents.extend(fallback_docs)
                print(f"Retrieved documents using fallback method: {len(fallback_docs)}")
        except Exception as fallback_error:
            print(f"Fallback query also failed: {fallback_error}")

    if not documents:
        print("No documents retrieved from any source")
                
    return {
        "document": documents, 
        "question": query, 
        "memory_context": state.get("memory_context", "")
    }


def evaluate_document(state: State) -> Dict[str, Any]:
    """Evaluate document relevance to the question."""
    print("---Check document relevant to question---")
    
    question = state['question']
    documents = state["document"]
    memory_context = state.get("memory_context", "")
    filtered_docs = []
    
    # Thêm kiểm tra cho trường hợp không có documents
    if not documents:
        print("No documents to evaluate")
        return {
            "document": [], 
            "question": question, 
            "web_search": "yes",  # Đề xuất web search khi không có tài liệu
            "memory_context": memory_context
        }
    
    for doc in documents:
        try:
            # Thêm thông tin để debug
            print(f"Evaluating document: {doc.page_content[:50]}...")
            
            score = retrieve_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            
            if score.binary_score.lower() == "yes":
                print("---Document relevant---")
                filtered_docs.append(doc)
            else:
                print(f"---Document not relevant--- Score: {score.binary_score}")
        except Exception as e:
            print(f"Error evaluating document: {e}")
            filtered_docs.append(doc)
            print("Added document despite evaluation error")
    
    total_docs = len(documents)
    relevant_docs = len(filtered_docs)
    
    if total_docs == 0 or (relevant_docs / total_docs) <= 0.5:
        web_search_needed = 'yes'
        print(f"Web search recommended: {relevant_docs}/{total_docs} relevant docs")
    elif relevant_docs >= 3:
        web_search_needed = 'no'
        print(f"Enough relevant docs ({relevant_docs}), no web search needed")
    # Tỉ lệ tương đối (>50% nhưng <3 tài liệu)
    else:
        web_search_needed = 'yes'
        print("Some relevant docs but not enough, web search recommended")
    
    return {
        "document": filtered_docs, 
        "question": question, 
        "web_search": web_search_needed,
        "memory_context": memory_context
    }


def update_history(state: State) -> Dict[str, Any]:
    """Update conversation history with the latest interaction"""
    question = state["question"]
    generation = state["generation"]
    
    # Parse the generation again to ensure it's clean
    # This is a safeguard in case the raw response somehow made it through
    clean_generation = parse_bot_response(generation)
    
    chat_history = state.get("chat_history", [])
    memory_context = state.get("memory_context", "")

    metadata = {
        "num_documents": len(state.get("document", [])),
        "web_search_used": state.get("web_search", "no"),
        "memory_used": bool(memory_context and memory_context != "No relevant past conversations found.")
    }

    chat_history.append({"user": question, "bot": clean_generation})

    memory.add_interaction(question, clean_generation, metadata)
    semantic_memory.add_memory([{
        "user": question,
        "bot": clean_generation, 
        "timestamp": datetime.now().isoformat(),
        "session_id": memory.session_id
    }])

    return {
        "document": state["document"],
        "question": question,
        "generation": clean_generation,  # Return the clean generation
        "chat_history": chat_history,
        "memory_context": memory_context
    }


def transform_query(state: State) -> Dict[str, Any]:
    """Transform the query to improve retrieval."""
    print("---Transform query---")
    
    question = state['question']
    documents = state['document']
    memory_context = state.get("memory_context", "")
    
    try:
        better_question = question_rewriter.invoke({"question": question})
        print(f"Transformed query: {better_question}")
    except Exception as e:
        print(f"Error transforming query: {e}")
        better_question = question
    
    return {
        "document": documents, 
        "question": better_question,
        "memory_context": memory_context
    }


def web_search(state: State) -> Dict[str, Any]:
    """Perform web search and add results to documents."""
    print("---WEB SEARCH---")
    
    question = state["question"]
    documents = state['document']
    memory_context = state.get("memory_context", "")
    
    try:
        search_results = web_search_tool.invoke({"query": question})
        if search_results:
            web_content = "\n\n".join([d.get('content', '') for d in search_results if 'content' in d])
            web_result = Document(page_content=web_content)
            documents.append(web_result)
            print(f"Added web search results ({len(search_results)} items)")
    except Exception as e:
        print(f"Error during web search: {e}")
    
    return {
        "document": documents, 
        "question": question,
        "memory_context": memory_context
    }


def decide_next_step(state: State) -> str:
    """Decide the next step in the pipeline."""
    print("---ASSESS DOCUMENT QUALITY---")
    
    web_search_needed = state.get("web_search", "").lower()
    
    if web_search_needed == "yes":
        print("--DECISION: DOCUMENTS NOT SUFFICIENTLY RELEVANT, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE RESPONSE---")
        return "generate"


def generate(state: State) -> Dict[str, Any]:
    """Generate a response based on the question, documents, memory and conversation history."""
    print('---GENERATE---')
    
    question = state["question"]
    documents = state["document"]
    chat_history = state.get("chat_history", [])
    memory_context = state.get("memory_context", "")
    
    # Format documents for context
    formatted_texts = []
    if documents and isinstance(documents, list):
        for doc in documents:
            if hasattr(doc, 'page_content'):
                formatted_texts.append(doc.page_content)
            elif isinstance(doc, str):
                formatted_texts.append(doc)
            elif isinstance(doc, dict) and "page_content" in doc:
                formatted_texts.append(doc["page_content"])
    
    context = "\n\n".join(formatted_texts) if formatted_texts else "No relevant documents found."
    
    # Format recent conversation history (last 3 interactions)
    history_context = ""
    if chat_history:
        history_context = "Previous conversation:\n"
        for interaction in chat_history[-3:]:
            if "user" in interaction and "bot" in interaction:
                history_context += f"User: {interaction['user']}\nAssistant: {interaction['bot']}\n\n"
    
    # Generate response with all context
    try:
        prompt = (
            f"Long-term memories from past conversations:\n{memory_context}\n\n"
            f"Recent conversation:\n{history_context}\n\n"
            f"Question: {question}\n\n"
            f"Context from documents: {context}\n\n"
            f"Answer:"
        )
        response = rag_llm.invoke(prompt)
        raw_response = str(response)
        
        # Parse the response to extract only the content
        generation_result = parse_bot_response(raw_response)
    except Exception as e:
        print(f"Error generating response: {e}")
        generation_result = "Sorry, I couldn't generate a response due to an error."
    
    return {
        "document": documents, 
        "question": question, 
        "generation": generation_result,  # Now contains only the clean content
        "chat_history": chat_history,
        "memory_context": memory_context
    }

def parse_bot_response(bot_response):

    if "content='" in bot_response:
        start_index = bot_response.find("content='") + 9
        end_index = bot_response.find("'", start_index)
        
        if start_index >= 9 and end_index > start_index:
            content = bot_response[start_index:end_index]
            content = content.replace("\\'", "'").replace("\\n", "\n").replace("\\\"", "\"")
            return content
    return bot_response