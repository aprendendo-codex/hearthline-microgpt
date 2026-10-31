# Generated from the shared Python trainer; package-local imports and model selection only.
"""Train from a UI-compatible match collection; NumPy is the only Python dependency."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
from hearthline import EngineClient
from learning import MicroGPT, Adam


def packed(rows):
    tokens = np.asarray([r["tokens"] for r in rows], dtype=np.int64)
    legal = np.full((len(rows), max(len(r["legal"]) for r in rows)), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        legal[i, :len(row["legal"])] = row["legal"]
    targets = np.asarray([r["target"] for r in rows], dtype=np.int64)
    return tokens, legal, targets


def train(input_file, output_file, name="Custom", epochs=1, width=16, heads=4,
          batch_size=16, learning_rate=0.003, seed=42, init_file=None, init_model=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
        raise ValueError("Use a model name of 1-64 letters, digits, dots, underscores or hyphens.")
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive.")
    raw = Path(input_file).read_bytes()
    collection = json.loads(raw)
    with EngineClient() as engine:
        info = engine.request("info")
        print("Encoding winner decisions...", file=sys.stderr, flush=True)
        data = engine.request("training_data", collection=collection)
        if not data["rows"]:
            raise ValueError("No decisive completed matches: draws and unfinished matches are skipped.")
        if init_file:
            envelope = json.loads(Path(init_file).read_text(encoding="utf-8"))
            entries = envelope.get("models", {})
            if init_model is None and len(entries) != 1:
                raise ValueError("Use --init-model to select a model from a multi-model file.")
            parent_name = init_model or next(iter(entries))
            if parent_name not in entries:
                raise ValueError("Unknown initialization model.")
            parent = entries[parent_name]
            engine.request("load_models", models=dict(format=1, models={parent_name: parent}))
            config = {k: parent["config"][k] for k in ("context", "vocab", "actions", "width", "heads", "seed")}
            model = MicroGPT(**config)
            for key, tensor in parent["weights"].items():
                model.p[key] = np.frombuffer(base64.b64decode(tensor["data"]), dtype="<f4").copy().reshape(tensor["shape"])
        else:
            if not 4 <= width <= 256 or heads < 1 or width % heads:
                raise ValueError("Width must be 4-256 and divisible by the positive head count.")
            model = MicroGPT(context=info["context"], vocab=info["vocab"], actions=info["actions"], width=width, heads=heads, seed=seed)
    tokens, legal, targets = packed(data.pop("rows"))
    count = len(tokens)
    rng = np.random.default_rng(seed)
    print("Training %d decisions from %d winning matches..." % (count, data["episodes"]), file=sys.stderr, flush=True)
    optimizer = Adam(model)
    for epoch in range(epochs):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, batch_size):
            ids = order[start:start + batch_size]
            loss, grads = model.loss(tokens[ids], legal[ids], targets[ids], backward=True)
            if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                raise ValueError("Training became non-finite; reduce the learning rate.")
            optimizer.update(model, grads, learning_rate)
            total += loss * len(ids)
        print("epoch %d/%d: loss %.6f (%d winner decisions)" % (epoch + 1, epochs, total / count, count), file=sys.stderr)
    digest = hashlib.sha256(json.dumps(model.config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    weights = {}
    for key in sorted(model.p):
        tensor = model.p[key].astype("<f4")
        if not np.isfinite(tensor).all():
            raise ValueError("Training produced non-finite weights.")
        binary = tensor.tobytes()
        digest.update(key.encode("ascii"))
        digest.update(binary)
        weights[key] = dict(shape=list(tensor.shape), data=base64.b64encode(binary).decode("ascii"))
    entry = dict(config=model.config, sha256=digest.hexdigest(), weights=weights,
                 trainingRulesVersion=data["rulesVersion"], encodingVersion=data["encodingVersion"],
                 trainingHeuristicVersion=data["heuristicVersion"], trainingParameters=data["parameters"],
                 datasetKind="winner-episodes", episodes=data["episodes"], decisions=count,
                 sampledDecisions=count * epochs, uniqueDecisionsSeen=count,
                 trainingDataSha256=hashlib.sha256(raw).hexdigest(),
                 parameters=sum(p.size for p in model.p.values()), benchmark=[])
    if init_file:
        entry.update(parentModel=parent_name, parentCheckpointSha256=parent["sha256"])
    result = dict(format=1, models={name:entry})
    # Use the same validator as the HTML before publishing a checkpoint.
    with EngineClient() as engine:
        engine.request("load_models", models=result)
    Path(output_file).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Match collection JSON exported by the UI or server")
    parser.add_argument("--output", default="new-model.json")
    parser.add_argument("--name", default="Custom")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init", help="Optional single-model JSON to fine-tune; keeps its architecture")
    parser.add_argument("--init-model", help="Model ID when --init contains several models")
    args = parser.parse_args()
    try:
        train(args.input, args.output, args.name, args.epochs, args.width, args.heads,
              args.batch_size, args.learning_rate, args.seed, args.init, args.init_model)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Training failed: " + str(error) + "\n")
    print("Saved UI-compatible model JSON.")


if __name__ == "__main__":
    main()
