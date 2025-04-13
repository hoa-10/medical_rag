import uuid
import os
from dotenv import load_dotenv
from function import State, retrieve, evaluate_document, generate, transform_query, web_search, decide_next_step, update_history, retrieve_memory
from conversationMemory import ConservationMemory
from Retriever_memory import SemanticMemoryRetriever
from langgraph.graph import START, END, StateGraph
from format import format_response_for_display

# Define the workflow
workflow = StateGraph(State)

# Add the nodes
workflow.add_node("retrieve_memory", retrieve_memory)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_document", evaluate_document)
workflow.add_node("web_search", web_search)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)
workflow.add_node("update_history", update_history)

# Define the edges
workflow.set_entry_point("retrieve_memory")
workflow.add_edge("retrieve_memory", "retrieve")
workflow.add_edge("retrieve", "grade_document")
workflow.add_edge("grade_document", "web_search")
workflow.add_edge("web_search", "transform_query")
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "update_history")

# Compile
app = workflow.compile()

def chat_with_rag():
    chat_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        initial_state = {
            "question": user_input,
            "chat_history": chat_history,
            "document": [],
            "generation": "",
            "web_search": "",
            "memory_context": ""
        }

        for output in app.stream(initial_state):
            for key, value in output.items():
                if key == "update_history":
                    print(f"Assistant: {value['generation']}")
                    chat_history = value["chat_history"]
        print()

if __name__ == "__main__":
    chat_with_rag() 