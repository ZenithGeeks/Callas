from orchestrator.show_graph import mermaid


def test_drawn_turn_graph_has_both_branches_after_decide():
    text = mermaid()
    assert "decide -.-> escalate;" in text
    assert "decide -.-> reply;" in text
    assert "reply --> speak;" in text
    assert "analyze --> verdict;" in text
