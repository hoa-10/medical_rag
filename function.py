from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
from pdf_retriever import process_all_pdf
import os
from agent import retrieve_grader, question_rewriter
from langchain.schema import Document
from langchain_tavily import TavilySearch
from conversationMemory import ConservationMemory
from Retriever_memory import SemanticMemoryRetriever
from datetime import datetime
from langgraph.constants import Send
from agent import rag_llm
from dotenv import load_dotenv

load_dotenv("api.env")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
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
    return str(question)


def retrieve_memory(state: State) -> Dict[str, Any]:
    """Retrieve relevant memories based on the question."""
    print("---RETRIEVING MEMORIES HISTORY---")
    
    query = _get_query_text(state["question"])
    memory_docs = semantic_memory.retrieve_relevant_memory(query, top_k=3)
    
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
    print("\n=== DOCUMENT RETRIEVAL PHASE ===")
    print("1. Processing query...")
    
    query = _get_query_text(state["question"])
    print(f"📝 Query: '{query}'")
    
    documents = []
    pdf_path = r"C:\Users\Admin\Desktop\medical_rag\folder_pdf"
    try:
        print("\n2. Searching PDF documents...")
        retriever = process_all_pdf(pdf_dir=pdf_path)
        main_docs = retriever.get_relevant_documents(query)
        documents.extend(main_docs)
        print(f"📚 Retrieved {len(main_docs)} documents from database")
        
        # Print brief preview of each document
        for i, doc in enumerate(main_docs, 1):
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"\nDocument {i}:")
            print(f"Preview: {preview}")
            
    except Exception as e:
        print(f"❌ Error retrieving from database: {e}")

    if not documents:
        print("⚠️ No documents retrieved from any source")
                
    return {
        "document": documents, 
        "question": query, 
        "memory_context": state.get("memory_context", "")
    }

#Evaluting the document
def evaluate_document(state: State) -> Dict[str, Any]:
    """Evaluate document relevance to the question."""
    print("\n=== DOCUMENT EVALUATION PHASE ===")
    
    question = state['question']
    documents = state["document"]
    memory_context = state.get("memory_context", "")
    filtered_docs = []
    
    print(f"📊 Evaluating {len(documents)} documents for relevance...")
    
    for i, doc in enumerate(documents, 1):
        try:
            print(f"\nAnalyzing Document {i}:")
            score = retrieve_grader.invoke({
                "question": question,
                "document": doc.page_content
            })
            
            if score.binary_score.lower() == "yes":
                print("✅ Document marked as RELEVANT")
                filtered_docs.append(doc)
            else:
                print("❌ Document marked as NOT RELEVANT")
        except Exception as e:
            print(f"Error evaluating document: {e}")
    
    # Calculate and display relevance statistics
    total_docs = len(documents)
    relevant_docs = len(filtered_docs)
    relevance_ratio = relevant_docs / total_docs if total_docs > 0 else 0
    print(f"\n📈 Relevance Statistics:")
    print(f"- Total Documents: {total_docs}")
    print(f"- Relevant Documents: {relevant_docs}")
    print(f"- Relevance Ratio: {relevance_ratio:.2%}")
    
    # Decide if web search is needed
    web_search_needed = 'yes' if documents and relevance_ratio <= 0.1 else 'no'
    print(f"\n🔍 Web Search Decision:")
    print(f"- Need web search: {web_search_needed.upper()}")
    if web_search_needed == 'yes':
        print("- Reason: Less than 50% of documents were relevant")
    
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
    chat_history = state.get("chat_history", [])
    memory_context = state.get("memory_context", "")

    metadata = {
        "num_documents": len(state.get("document", [])),
        "web_search_used": state.get("web_search", "no"),
        "memory_used": bool(memory_context and memory_context != "No relevant past conversations found.")
    }

    chat_history.append({"user": question, "bot": generation})

    memory.add_interaction(question, generation, metadata)
    semantic_memory.add_memory([{
        "user": question,
        "bot": generation, 
        "timestamp": datetime.now().isoformat(),
        "session_id": memory.session_id
    }])

    return {
        "document": state["document"],
        "question": question,
        "generation": generation,
        "chat_history": chat_history,
        "memory_context": memory_context
    }


def transform_query(state: State) -> Dict[str, Any]:
    """Transform the query to improve retrieval."""
    print("\n=== QUERY TRANSFORMATION PHASE ===")
    
    question = state['question']
    documents = state['document']
    memory_context = state.get("memory_context", "")
    
    print(f"Original query: '{question}'")
    
    try:
        better_question = question_rewriter.invoke({"question": question})
        print(f"✨ Transformed query: '{better_question}'")
    except Exception as e:
        print(f"❌ Error transforming query: {e}")
        print("Using original query instead")
        better_question = question
    
    return {
        "document": documents, 
        "question": better_question,
        "memory_context": memory_context
    }


def web_search(state: State) -> Dict[str, Any]:
    """Perform web search and add results to documents."""
    print("\n=== WEB SEARCH PHASE ===")
    question = state["question"]
    documents = state['document']
    memory_context = state.get("memory_context", "")
    
    print(f"🌐 Searching web for: '{question}'")
    try:
        search_results = web_search_tool.invoke({"query": question})
        
        if search_results:
            # Process web content for documents
            web_contents = []
            for d in search_results:
                if isinstance(d, dict) and 'content' in d:
                    web_contents.append(d['content'])
            
            if web_contents:
                web_content = "\n\n".join(web_contents)
                web_result = Document(page_content=web_content)
                documents.append(web_result)
                print(f"✅ Added {len(web_contents)} web search results")
            
            # Preview web results - separate try block to isolate any issues
            try:
                print("\nWeb Search Results Preview:")
                preview_count = min(3, len(search_results))
                
                for i in range(preview_count):
                    result = search_results[i]
                    if isinstance(result, dict) and 'content' in result:
                        content = result['content']
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"{i+1}. {preview}")
            except Exception as e:
                print(f"⚠️ Error during preview generation: {str(e)}")
                
    except Exception as e:
        print(f"❌ Error during web search: {e}")
    
    return {
        "document": documents,
        "question": question,
        "memory_context": memory_context
    }

def decide_next_step(state: State) -> str:
    """Decide the next step in the pipeline."""
    print("\n=== DECISION PHASE ===")
    
    web_search_needed = state.get("web_search", "").lower()
    
    print("🤔 Evaluating next step...")
    if web_search_needed == "yes":
        print("Decision: Documents not sufficiently relevant")
        print("Action: Will transform query and perform web search")
        return "transform_query"
    else:
        print("Decision: Sufficient relevant documents found")
        print("Action: Proceeding to generate response")
        return "generate"


def generate(state: State) -> Dict[str, Any]:
    """Generate a response based on the question, documents, memory and conversation history."""
    print("\n=== RESPONSE GENERATION PHASE ===")
    
    question = state["question"]
    documents = state["document"]
    chat_history = state.get("chat_history", [])
    memory_context = state.get("memory_context", "")
    
    print("1. Preparing Context...")
    
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
    print(f"📚 Using {len(formatted_texts)} documents as context")
    
    # Format recent conversation history
    history_context = ""
    if chat_history:
        history_context = "Previous conversation:\n"
        recent_history = chat_history[-3:]  # Last 3 interactions
        print(f"💭 Including {len(recent_history)} recent conversation turns")
        for interaction in recent_history:
            if "user" in interaction and "bot" in interaction:
                history_context += f"User: {interaction['user']}\nAssistant: {interaction['bot']}\n\n"
    
    print("\n2. Generating Response...")
    try:
        prompt = (
            f"Long-term memories from past conversations:\n{memory_context}\n\n"
            f"Recent conversation:\n{history_context}\n\n"
            f"Question: {question}\n\n"
            f"Context from documents: {context}\n\n"
            f"Answer:"
        )
        response = rag_llm.invoke(prompt)
        generation_result = str(response)
        print("✅ Response generated successfully")
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        generation_result = "Sorry, I couldn't generate a response due to an error."
    
    return {
        "document": documents, 
        "question": question, 
        "generation": generation_result,
        "chat_history": chat_history,
        "memory_context": memory_context
    }