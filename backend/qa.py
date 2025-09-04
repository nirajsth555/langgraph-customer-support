from graph_pipeline import run_graph

def ask_question(question: str, k: int = 3) -> str:
    return run_graph(question)
