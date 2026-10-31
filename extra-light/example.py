"""Generate seeded headless games, retain decisive matches, then train a policy."""
import argparse
import json
from pathlib import Path

import sys
sys.dont_write_bytecode = True
from hearthline import EngineClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "example")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--width", type=int, default=16)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("Generate at least one game.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches, winners = [], []
    policies = ["balanced", "rush", "economy"]
    with EngineClient() as engine:
        for i in range(args.games):
            pair = [policies[i % 3], policies[(i // 3 + i + 1) % 3]]
            game = engine.request("simulate", config=dict(mapSeed="example-map:%d:%d" % (args.seed, i),
                                  aiSeed="example-ai:%d:%d" % (args.seed, i), policies=pair))
            entry = game["collection"]["matches"][0]
            matches.append(entry)
            if game["result"]["winner"] is not None:
                winners.append(entry)
            print("game %d/%d: %s" % (i + 1, args.games, game["result"]["reason"]))
    def save(name, entries):
        data = dict(format="hearthline-matches", version=1, meta=dict(seed=str(args.seed)), matches=entries)
        target = args.output_dir / name
        target.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        return target
    save("matches.json", matches)
    if not winners:
        parser.exit(1, "No decisive games; try more games or another seed. matches.json was retained.\n")
    training_file = save("winners.json", winners)
    from train import train
    train(training_file, args.output_dir / "new-model.json", name="ExternalDemo", epochs=args.epochs, width=args.width)
    print("Import matches.json or winners.json in Match archive; import new-model.json in Trained models.")


if __name__ == "__main__":
    main()
