"""The wrappers must turn the raw outputs of transformers / gliner into our dict format."""
import pytest

from masker.detectors.model_detectors import GLiNERDetector, HFTokenDetector


class FakePipe:
    def __call__(self, texts, batch_size=16):
        return [[{"entity_group": "PER", "score": 0.99, "word": "Barack Obama", "start": 0, "end": 12}]
                for _ in texts]


class NewGliner:  # gliner >= 0.2 API
    def inference(self, texts, labels, threshold=0.5, batch_size=8):
        return [[{"start": 0, "end": 3, "text": "New", "label": "location", "score": 0.9},
                 {"start": 4, "end": 8, "text": "York", "label": "location", "score": 0.8}] for _ in texts]


class OldGliner:  # older API
    def batch_predict_entities(self, texts, labels, threshold=0.5):
        return [[{"start": 0, "end": 8, "text": "New York", "label": "location", "score": 0.7}] for _ in texts]


def test_hf_wrapper():
    d = HFTokenDetector("x", pipe=FakePipe(), device="cpu")
    assert d.predict(["Barack Obama met"]) == [[{"start": 0, "end": 12, "label": "PER", "score": 0.99,
                                                 "text": "Barack Obama"}]]


def test_gliner_wrapper_merges_token_level_fragments():
    d = GLiNERDetector("x", merge=True, lowercase_labels=True, model=NewGliner(), device="cpu")
    out = d.predict(["New York is big"], ["Location"])
    assert out == [[{"start": 0, "end": 8, "label": "location", "score": 0.8, "text": "New York"}]]


def test_gliner_wrapper_old_api():
    d = GLiNERDetector("x", model=OldGliner(), device="cpu")
    assert d.predict(["New York"], ["location"])[0][0]["text"] == "New York"


def test_real_transformers_pipeline_output_format():
    """Build a tiny random BERT offline and run the real HF pipeline through the wrapper."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import tempfile, os

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "barack", "obama", "met", "in", "baghdad", "."]
    with tempfile.TemporaryDirectory() as tmp:
        vocab = os.path.join(tmp, "vocab.txt")
        with open(vocab, "w") as fh:
            fh.write("\n".join(words))
        tok = transformers.BertTokenizerFast(vocab_file=vocab, do_lower_case=True)
        labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        cfg = transformers.BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                      num_attention_heads=2, intermediate_size=32, num_labels=len(labels),
                                      id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
        torch.manual_seed(0)
        model = transformers.BertForTokenClassification(cfg).eval()
        pipe = transformers.pipeline("token-classification", model=model, tokenizer=tok,
                                     aggregation_strategy="max", device="cpu")
        det = HFTokenDetector("tiny", pipe=pipe, device="cpu")
        texts = ["Barack Obama met in Baghdad.", "Obama."]
        out = det.predict(texts)
        assert len(out) == 2
        for text, ents in zip(texts, out):
            for e in ents:
                assert set(e) == {"start", "end", "label", "score", "text"}
                assert e["label"] in {"PER", "LOC"} and text[e["start"]:e["end"]] == e["text"]
