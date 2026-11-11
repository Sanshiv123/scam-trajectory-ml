"""Content-addressed frozen transformer inference; downloads happen only on cache miss."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
EMOTION = "j-hartmann/emotion-english-distilroberta-base"
TEXT = "distilbert/distilbert-base-uncased"
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def encode(texts, kind="emotion", batch_size=16, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    model_id = EMOTION if kind == "emotion" else TEXT
    folder = ROOT / ".cache" / kind
    folder.mkdir(parents=True, exist_ok=True)
    index_path, info_path = folder / "vectors.npz", folder / "model.json"
    cache = dict(np.load(index_path)) if index_path.exists() else {}
    key = lambda t: hashlib.sha256(f"{model_id}|{max_length}|{t}".encode()).hexdigest()
    missing = {key(t): t for t in texts if key(t) not in cache}
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if missing:
        print(f"{kind}: encoding {len(missing)} unique uncached texts", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=info.get("revision", "main"))
        cls = AutoModelForSequenceClassification if kind == "emotion" else AutoModel
        model = cls.from_pretrained(model_id, revision=info.get("revision", "main")).eval()
        if info and info["revision"] != model.config._commit_hash:
            raise ValueError("Model revision changed within cache")
        items = list(missing.items())
        truncated = 0
        for start in range(0, len(items), batch_size):
            chunk = items[start:start+batch_size]
            raw = tokenizer([t for _, t in chunk], add_special_tokens=True, truncation=False)
            truncated += sum(len(ids) > max_length for ids in raw["input_ids"])
            inputs = tokenizer([t for _, t in chunk], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
            with torch.inference_mode():
                output = model(**inputs)
                if kind == "emotion":
                    probs = output.logits.softmax(-1).numpy()
                    indices = [next(int(i) for i, name in model.config.id2label.items() if name == lab) for lab in LABELS]
                    vectors = probs[:, indices]
                else:
                    # Frozen token-mean representation, not fine-tuned DistilBERT.
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    vectors = ((output.last_hidden_state * mask).sum(1) / mask.sum(1)).numpy()
            for (digest, _), vec in zip(chunk, vectors): cache[digest] = vec
            if start % (batch_size*25) == 0:
                print(f"{kind}: {min(start+batch_size,len(items))}/{len(items)}", flush=True)
        np.savez_compressed(index_path, **cache)
        info = dict(model=model_id, revision=model.config._commit_hash, max_length=max_length,
                    cached_texts=len(cache), truncated_texts=info.get("truncated_texts", 0)+truncated,
                    inference="CPU float32 eval, fixed batch size, deterministic torch")
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        del model
    return np.stack([cache[key(t)] for t in texts]), info
