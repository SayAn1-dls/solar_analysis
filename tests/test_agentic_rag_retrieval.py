from src.agentic_rag.retrieval import GuidelineRetriever


def test_retrieval_returns_relevant_guidance():
    retriever = GuidelineRetriever()
    results = retriever.search("battery discharge during solar deficit and reserve margin", top_k=3)

    assert len(results) == 3
    assert any("Battery" in item["title"] or "Grid" in item["title"] for item in results)
