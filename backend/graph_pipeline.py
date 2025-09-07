from typing import List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import MessagesState
from ingestion import load_vectorstore, build_hybrid_retriever, retrieve_context, prompting
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

memory = MemorySaver()
graph = None

def get_graph():
    global graph
    if graph is None:
        graph = StateGraph(QAState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("prompt", prompt_node)
        graph.add_node("llm", llm_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "prompt")
        graph.add_edge("prompt", "llm")
        graph.add_edge("llm", END)

        # attach memory
        graph = graph.compile(checkpointer=memory)
    return graph

# ----------------------------
# Step 1: Define State Schema
# ----------------------------
class QAState(MessagesState):
    """
    State passed between nodes.
    Includes chat history + extra fields.
    """
    context: List[Document]
    prompt: str
    answer: str

# ----------------------------
# Step 2: Define Nodes
# ----------------------------
def retrieve_node(state: QAState) -> QAState:
    """Retrieve documents from vectorstore using hybrid retriever."""
    question = state["messages"][-1].content  # last human msg = question

    vs = load_vectorstore()
    retriever = build_hybrid_retriever(vs)
    docs = retrieve_context(retriever, question, k=3)

    state["context"] = docs
    return state


def prompt_node(state: QAState) -> QAState:
    """Format context + question into a structured prompt."""
    question = state["messages"][-1].content
    docs = state.get("context", [])
    state["prompt"] = prompting(docs, question)
    return state


def llm_node(state: QAState) -> QAState:
    """Send prompt to LLM and return answer."""
    chat = init_chat_model(model="gpt-4.1-nano", api_key=OPENAI_API_KEY)

    response = chat.invoke([HumanMessage(content=state["prompt"])])
    return {
        "messages": [AIMessage(content=response.content)],
        "answer": response.content,
    }

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

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# ----------------------------
# Step 4: Public Run Function
# ----------------------------
def run_graph(question: str, thread_id: str = "default") -> str:
    # graph = build_graph()
    graph = get_graph()

    # Start with a HumanMessage for the user question
    inputs = {"messages": [HumanMessage(content=question)]}

    result = graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
    for msg in result["messages"]:
        print(msg.type, ":", msg.content)
    return result.get("answer")
