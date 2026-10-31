# Hearthline

A tiny turn-based strategy game in a single HTML file, where AI players are driven by heuristics or small [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)-style models.

Two factions share a mirrored 8×8 board. Each turn both sides secretly issue up to three commands, then they resolve at the same time. Workers gather wood and ore and tend farms. Soldiers march on the enemy hearth. Every unit eats food. You win by destroying the enemy hearth or starving them out, and after 80 turns the game is a draw.

## Play

Open `hearthline.html` in a browser: https://aprendendo-codex.github.io/hearthline.html

It needs no install or server. You can:

- play against an AI;
- watch AI vs AI matches, replay them and view stats;
- run batches in the simulation lab and tune the game balance;
- import and export matches and trained models as JSON.

Built-in AIs: **Steward** (balanced), **Raider** (rush), **Cultivator** (economy) and the trained models **V1–V5**.

## extra-light: Python SDK

[`extra-light/`](extra-light) is a native Python version of the rules for headless matches and model training. It uses only the standard library, plus NumPy for models.

```bash
cd extra-light
python -m pip install -r requirements.txt
python example.py --games 20     # simulate games and train a model from the winners
```

| File | Purpose |
| --- | --- |
| `hearthline.py` | Rules engine, built-in AIs, in-process API and JSON-lines server |
| `learning.py` | NumPy MicroGPT and Adam optimizer |
| `train.py` | Train or fine-tune a model from a match file |
| `models.json` | Bundled V1–V5 weights |

You can use it to train your own models. Matches and models you produce can be imported into the HTML game. See [`extra-light/README.md`](extra-light/README.md) for the full protocol.
