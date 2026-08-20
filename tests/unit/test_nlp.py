from rookie_lab.nlp.tokenizer import LocalTokenizer


def test_local_tokenizer_round_trip() -> None:
    tokenizer = LocalTokenizer(["helpful service", "late delivery"], max_length=8)
    ids, mask = tokenizer.batch(["helpful delivery"])
    assert ids.shape == (1, 8)
    assert mask.sum() == 4
    assert "helpful" in tokenizer.decode(ids[0].tolist())
