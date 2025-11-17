from pathlib import Path
import networkx as nx


def build_causal_graph():
    G = nx.DiGraph()
    # Nodes
    G.add_node("user")
    G.add_node("preference")
    G.add_node("news_features")
    G.add_node("category")
    G.add_node("recommendation")
    G.add_node("click")

    # Edges representing assumptions
    G.add_edge("user", "preference")
    G.add_edge("preference", "click")
    G.add_edge("news_features", "click")
    G.add_edge("category", "click")
    G.add_edge("recommendation", "click")

    return G


if __name__ == "__main__":
    g = build_causal_graph()
    print("Causal graph built with nodes:", g.nodes())
