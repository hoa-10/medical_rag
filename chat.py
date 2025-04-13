import uuid
import os
from dotenv import load_dotenv
from function import State, retrieve, evaluate_document, generate, transform_query, web_search, decide_next_step, update_history, retrieve_memory
from conversationMemory import ConservationMemory
from Retriever_memory import SemanticMemoryRetriever
from langgraph.graph import START, END, StateGraph
from format import format_response_for_display

#HELLOOOOOOOOOOOOOOOOOOOOO
memory = ConservationMemory()
semantic_memory = SemanticMemoryRetriever()
semantic_memory.add_memory(memory.get_all_history())

workflow = StateGraph(State)


workflow.add_node("retrieve_memory", retrieve_memory)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_document", evaluate_document)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_tool", web_search)
workflow.add_node("update_history", update_history)


workflow.add_edge(START, "retrieve_memory")
workflow.add_edge("retrieve_memory", "retrieve")
workflow.add_edge("retrieve", "grade_document")
workflow.add_conditional_edges(
    "grade_document",
    decide_next_step,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)
workflow.add_edge("transform_query", "web_search_tool")
workflow.add_edge("web_search_tool", "generate")
workflow.add_edge("generate", "update_history")
workflow.add_edge("update_history", END)

app = workflow.compile()

def chat_with_rag():
    print("Chat with the Memory-Enhanced RAG system (type 'exit' to quit, 'clear' to clear current session)")
    chat_history = memory.get_recent_history()
    
    print(f"Loaded {len(memory.get_all_history())} previous interactions from memory")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            memory.clear_current_session()
            chat_history = []
            print("Current session cleared.")
            continue
        
        initial_state = {
            "question": user_input,
            "chat_history": chat_history,
            "document": [],  
            "generation": "",  
            "web_search": "no",  
            "memory_context": ""  
        }
        
        for output in app.stream(initial_state):
            for key, value in output.items():
                if key == "update_history":
                    chat_history = value.get("chat_history", [])
                    print(f"Assistant: {value.get('generation', '')}")
        print()

if __name__ == "__main__":
    chat_with_rag() 