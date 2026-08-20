import torch

from rookie_lab.nlp.experiment import TextRecurrentClassifier
from rookie_lab.nlp.tokenizer import LocalTokenizer


def test_local_tokenizer_round_trip() -> None:
    tokenizer = LocalTokenizer(["helpful service", "late delivery"], max_length=8)
    ids, mask = tokenizer.batch(["helpful delivery"])
    assert ids.shape == (1, 8)
    assert mask.sum() == 4
    assert "helpful" in tokenizer.decode(ids[0].tolist())


def test_unknown_tokens_map_to_unk() -> None:
    tokenizer = LocalTokenizer(["known words"], max_length=6)
    encoded = tokenizer.encode("unseen")
    assert tokenizer.vocab["[UNK]"] in encoded


def test_text_recurrent_model_shapes() -> None:
    ids = torch.ones(3, 8, dtype=torch.long)
    for cell in ("lstm", "gru"):
        assert TextRecurrentClassifier(20, cell)(ids).shape == (3, 2)
