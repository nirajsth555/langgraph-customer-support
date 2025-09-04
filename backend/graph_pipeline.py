from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from typing import List, Dict, Any
from ingestion import load_vectorstore, build_hybrid_retriever, retrieve_context, prompting
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Step 1: Define Graph State
# ----------------------------
class QAState(Dict[str, Any]):
    question: str
    context: List[Document]
    prompt: str
    answer: str

# ----------------------------
# Step 2: Define Nodes
# ----------------------------
def retrieve_node(state: QAState) -> QAState:
    """Retrieve documents from vectorstore using hybrid retriever."""
    vs = load_vectorstore()
    retriever = build_hybrid_retriever(vs)
    docs = retrieve_context(retriever, state["question"], k=3)
    state["context"] = docs
    return state

def prompt_node(state: QAState) -> QAState:
    """Format context + question into a structured prompt."""
    state["prompt"] = prompting(state["context"], state["question"])
    return state

def llm_node(state: QAState) -> QAState:
    """Send prompt to LLM and return answer."""
    chat = init_chat_model(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)
    response = chat.invoke([{"role": "user", "content": state["prompt"] }])
    state["answer"] = response.content
    return state

# ----------------------------
# Step 3: Assemble Graph
# ----------------------------
def build_graph():
    graph = StateGraph(QAState)

    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("prompt", prompt_node)
    graph.add_node("llm", llm_node)

    # Set entrypoint and edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "prompt")
    graph.add_edge("prompt", "llm")
    graph.add_edge("llm", END)
    
    
    # Define flow (functions are fine here!)
    # graph.add_sequence([retrieve_node, prompt_node, llm_node])

    # Start at retrieve
    # graph.add_edge(START, "retrieve")

    return graph.compile()

# ----------------------------
# Step 4: Public Run Function
# ----------------------------
def run_graph(question: str) -> str:
    graph = build_graph()
    result = graph.invoke({"question": question})
    print('result', result)
    return result.get("answer")
