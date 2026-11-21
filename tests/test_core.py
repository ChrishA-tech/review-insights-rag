import pandas as pd

from rvi import agent, aspects, retrieval


def test_aspect_rules():
    df = pd.DataFrame({"title": ["", "", ""], "text": ["This runs small, size up.", "Fabric is thin and see-through.",
                                                       "So soft and comfy!"]})
    t = aspects.tag(df)
    assert t.loc[0, "runs_small"] == 1 and t.loc[0, "runs_large"] == 0
    assert t.loc[1, "fabric_quality"] == 1 and t.loc[2, "comfort"] == 1


def test_bm25_finds_exact_review():
    texts = ["the zipper broke after one wash", "lovely soft cotton tee", "dress runs very large in the chest"]
    idx = retrieval.Index(texts, dims=2, min_df=1)
    assert idx.search("zipper broke", k=1, method="bm25") == [0]
    assert idx.search("runs large chest", k=1, method="hybrid") == [2]


def test_citation_and_quote_checks():
    ev = [{"reviews": [{"review_id": 42, "text": "The fabric pilled after two washes."}]}]
    assert agent.bad_citations('Customers say "fabric pilled after two washes" [42].', ev) == []
    assert agent.bad_citations("See [999].", ev) == ["[999]"]
    assert agent.bad_citations('One said "it shrank badly" [42].', ev) != []


def test_citation_numbers_are_not_treated_as_statistics():
    assert agent.unsupported("Reviews [1234] and [5678] mention it.", [{"reviews": []}], "q") == []


def test_citation_check_ignores_non_review_tool_outputs():
    ev = [{"reviews": 22621, "aspect_rates": {}}, {"reviews": [{"review_id": 7, "text": "Runs small."}]}]
    assert agent.bad_citations("See [7].", ev) == []


def test_citation_formats_are_all_checked():
    ev = [{"reviews": [{"review_id": 15119, "text": "So comfortable and soft."}, {"review_id": 3480, "text": "Love these."}]}]
    assert agent.cited_ids("Liked [15119, 3480].") == [15119, 3480]
    assert agent.bad_citations("Liked [15119, 9999].", ev) == ["[9999]"]
    assert agent.bad_citations("Review ID 9999 says so.", ev) == ["[9999]"]
    assert agent.bad_citations('The aspect "runs_large" matters.', ev) == []  # a label, not a quotation
