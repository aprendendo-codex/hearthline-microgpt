# Hearthline extra-light

A flat, native Python implementation of Hearthline's rules and headless tools. There is no Node.js dependency, embedded JavaScript, external process or network service. Heuristic simulation uses only Python's standard library. NumPy is needed only for trained-model inference and training.

Requires Python 3.10+. For all features, install the one optional package:

```text
python -m pip install -r requirements.txt
python example.py --games 20 --seed 2026
```

Run commands from this folder. The example generates seeded games, keeps decisive matches and trains a new model. Its `runs/example/matches.json` and `winners.json` import through the HTML UI's **Match archive → Import JSON**. `new-model.json` imports through **Trained models**. Losing-side commands stay in the saved match so it remains replayable; only the actual winner's decisions become training targets. Draws and partial games are skipped for training.

## Small file set

- `hearthline.py`: native rules, map/seed logic, heuristic controllers, observation/action encoding, match/model validation, in-process API and stdio server. This is maintained Python, not generated JavaScript or a translated runtime payload.
- `learning.py`: the shared NumPy MicroGPT/Adam implementation.
- `train.py`: training from a UI match collection, with fresh or supplied weights.
- `example.py`: generation → decisive-game filtering → training.
- `models.json`: exact V1–V5 weights in one UI-compatible envelope, reducing separate files. To export one: `python hearthline.py --export-model V4 --output V4.json`.
- `requirements.txt`, `llms.txt` and this guide; `MANIFEST.json` contains relative file hashes. Generated runs/caches/environments are ignored.

The native implementation preserves simultaneous sealed orders, queued jobs, transport, recurring/exclusive farm assignments, persistent movement, traffic, combat, recruitment, upkeep and victories. Deterministic map generation and tie-breaking use the same seed specification as the HTML. Recorded matches are replayed from orders, without requiring their original models. The repository checks native states, events, legal queues and encoding against the canonical engine; future rule changes require reviewing this Python implementation.

## Train or fine-tune

```text
python train.py matches.json --name Custom --output custom.json --width 16 --epochs 1
python train.py matches.json --name Refined --output refined.json --init models.json --init-model V4
```

Options also include `--heads`, `--batch-size`, `--learning-rate` and `--seed`. Initialization keeps the selected model's architecture. A multi-model file requires `--init-model`; a one-model export does not. All used matches must share one parameter snapshot. Training uses friendly-side normalization, legally masked sequential decisions and explicit stop targets. Output metadata records real sample counts and content hashes, with no invented benchmark scores or input paths. Check performance on different seeds; imitation learning does not guarantee stronger play.

## Use the API or stdio

```python
import json
from hearthline import EngineClient

with EngineClient() as game:
    result = game.request("simulate", config={
        "mapSeed": "demo-map", "aiSeed": "demo-ai",
        "policies": ["balanced", "rush"]})
    with open("matches.json", "w", encoding="utf-8") as output:
        json.dump(result["collection"], output)
```

Despite the matching API name, `EngineClient` is entirely in-process Python. No executable is discovered or started. Heuristic policies are `balanced` (Steward), `rush` (Raider), `economy` (Cultivator). Trained policies are `V1`–`V5` and imported IDs; these lazily load NumPy. To use another agent, `start` with `policies:["external","rush"]`, query `legal` for side 0 and your own `prefix`, then `step` with `orders:[prefix,null]` (Python `None`). Null asks the other side's AI to choose privately; an empty list passes. A wait action/ID 0 stops your command list: submit the existing prefix without appending wait to other commands.

`python hearthline.py` exposes the same operations over UTF-8 JSON Lines. Simulation works even with site packages disabled: `python -S hearthline.py`. One request yields one response, and malformed requests do not replace the active match.

```json
{"id":1,"op":"simulate","config":{"mapSeed":"demo","policies":["balanced","rush"]}}
```

Responses are `{id,ok:true,data:...}` or `{id,ok:false,error:...}`. Save `data.collection`, not this transport wrapper.

| Operation | Fields | Data |
| --- | --- | --- |
| `info` | none | versions, dimensions, default settings, policy IDs |
| `simulate` | `config` | one-match collection, result, turns |
| `start` | `config` (external sides allowed) | public state/result/turn |
| `state` | none | current state/result/turn |
| `legal` | `side` 0/1, `prefix` default [] | normalized observation, tokens, aligned legal IDs/actions, recovery action |
| `step` | `orders` [list or null, list or null] | simultaneous next state/result/turn |
| `export` | none | active complete or partial collection |
| `load_models` | `models` envelope | registered IDs (including collision renames) |
| `export_models` | optional `ids` list | model envelope under `models` |
| `training_data` | `collection` | actual-winner examples and training settings/counts |

Config supports `mapSeed`, `aiSeed`, two `policies`, optional numeric `parameters`, two `playerSeeds`/`seeds`, and two optional `bases`. Query `info` for settings. Legal actions already address the original engine board; do not rotate them again. Query legal choices again after adding each command to reserve resources, sites and units. The default allows three commands per side, one per unit. Both sides' decisions always refer to the same pre-turn state. The shared wood-recovery action is also applied to trained policies.

## UI compatibility and privacy

Matches use `{format:"hearthline-matches",version:1,meta:{},matches:[{config,initial,orders,entrants}]}`. A single match is a collection of one. Results/frames are reconstructed from the initial state and both order lists. Models use `{format:1,models:{Name:entry}}`, with base64 little-endian float32 tensors and matching rules/encoding. Import current UI match exports directly; the older single-replay envelope is also readable. Configuration-only legacy batches need to be converted through the main UI first.

Shipping files contain no account details, credentials, device identifiers or hard-coded machine/home paths. Game seeds/settings are caller-supplied data; no environment metadata is added to exports. The package is self-contained. The repository's `npm run build:light` (also part of its main build) refreshes combined weights and shared Python training helpers, verifies native version markers, emits the manifest and scans public files for private information. Node.js is used only by repository build/parity tests, never by this distributed package. `hearthline.py` is maintained directly and is not overwritten by the build.
