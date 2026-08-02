import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def test_tiny_hotpotqa_fixture_has_valid_two_hop_examples() -> None:
    examples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert all(
        set(example) == {"_id", "question", "answer", "context", "supporting_facts"}
        for example in examples
    )
    assert [example["_id"] for example in examples] == ["needle-001", "needle-002", "needle-003"]
    assert [example["answer"] for example in examples] == [
        "United States",
        "Pacific Ocean",
        "Quito",
    ]
    assert [[title for title, _ in example["context"]] for example in examples] == [
        ["The Left Hand of Darkness", "Ursula K. Le Guin", "Berkeley"],
        ["Machu Picchu", "Peru", "Atlantic Ocean"],
        ["Galapagos Islands", "Ecuador", "Guayaquil"],
    ]
    assert [example["supporting_facts"] for example in examples] == [
        [["The Left Hand of Darkness", 1], ["Ursula K. Le Guin", 1]],
        [["Machu Picchu", 1], ["Peru", 1]],
        [["Galapagos Islands", 1], ["Ecuador", 1]],
    ]
    expected_evidence = {
        "needle-001": {
            ("The Left Hand of Darkness", 1): "It was written by Ursula K. Le Guin.",
            ("Ursula K. Le Guin", 1): "She was born in Berkeley, California, in the United States.",
        },
        "needle-002": {
            ("Machu Picchu", 1): "It is located in Peru.",
            ("Peru", 1): "Peru has a coastline on the Pacific Ocean.",
        },
        "needle-003": {
            ("Galapagos Islands", 1): "They are part of Ecuador.",
            ("Ecuador", 1): "Its capital city is Quito.",
        },
    }

    for example in examples:
        context = example["context"]
        titles = [title for title, _ in context]
        supporting_facts = example["supporting_facts"]

        assert len(context) == 3
        assert len(titles) == len(set(titles))
        assert all(len(sentences) == 2 and all(sentences) for _, sentences in context)
        assert len(supporting_facts) == 2

        supporting_titles = {title for title, _ in supporting_facts}
        assert len(supporting_titles) == 2
        assert supporting_titles < set(titles)
        assert len(set(titles) - supporting_titles) == 1

        sentences_by_title = {title: sentences for title, sentences in context}
        for title, index in supporting_facts:
            assert 0 <= index < len(sentences_by_title[title])
        assert {
            (title, index): sentences_by_title[title][index] for title, index in supporting_facts
        } == expected_evidence[example["_id"]]
