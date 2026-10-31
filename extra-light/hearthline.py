"""Native Python Hearthline rules, headless JSONL server and in-process client.

Simulation/heuristics use only the standard library. NumPy is loaded only for
trained policies; no subprocess, JavaScript interpreter or network is used.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
RULES_VERSION, HEURISTIC_VERSION, ENCODING_VERSION = 7, 4, 2
CONTEXT, VOCAB, ACTIONS = 537, 130, 2083
DEFAULTS = {"workerHP":6,"workerDamage":1,"workerCost":{"wood":3,"ore":0,"food":3},"workerTime":2,"workerUpkeep":2,
 "soldierHP":12,"soldierDamage":3,"soldierCost":{"wood":3,"ore":5,"food":5},"soldierTime":4,"soldierUpkeep":3,
 "woodLoad":6,"woodGatherTime":2,"oreLoad":7,"oreGatherTime":3,"harvest":12,"growTime":4,"farmCost":{"wood":2},
 "renewCost":{"wood":1},"renewTime":1,"initial":{"wood":20,"ore":5,"food":12},"initialWorkers":2,"initialSoldiers":0,
 "baseHP":40,"baseDefense":1,"wallHP":12,"wallCost":{"wood":3,"ore":1},"buildTime":2,"commandsPerTurn":3,
 "cap":8,"tileCap":4,"maxTurns":80,"starvationLimit":3}
BOUNDS = {}
for _kind in ("worker", "soldier"):
    for _key, _lo, _hi in (("HP",1,200),("Damage",0,50),("Time",1,40),("Upkeep",0,100)):
        BOUNDS[_kind+_key] = (_lo,_hi)
    for _r in ("wood","ore","food"):
        BOUNDS[_kind+"Cost."+_r] = (0,100)
for _r in ("wood","ore"):
    BOUNDS[_r+"Load"], BOUNDS[_r+"GatherTime"] = (1,100),(1,40)
BOUNDS.update({"harvest":(1,100),"growTime":(1,40),"farmCost.wood":(0,100),"renewCost.wood":(0,100),
 "renewTime":(1,40),"initialWorkers":(1,4),"initialSoldiers":(0,3),"baseHP":(1,300),"baseDefense":(0,50),
 "wallHP":(1,200),"wallCost.wood":(0,100),"wallCost.ore":(0,100),"buildTime":(1,40),"commandsPerTurn":(1,3),
 "cap":(1,8),"tileCap":(1,4),"maxTurns":(10,300),"starvationLimit":(1,20)})
for _r in ("wood","ore","food"):
    BOUNDS["initial."+_r] = (0,500)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def normalize_parameters(value=None):
    value = {} if value is None else value
    if not isinstance(value, dict):
        raise ValueError("Parameters must be an object.")
    out = deepcopy(DEFAULTS)
    def visit(obj, prefix=""):
        for key, val in obj.items():
            name = prefix + key
            if name in BOUNDS:
                lo, hi = BOUNDS[name]
                if type(val) is not int or not lo <= val <= hi:
                    raise ValueError(name + " is outside its integer range.")
                if "." in name:
                    parent, child = name.split(".")
                    out[parent][child] = val
                else:
                    out[name] = val
            elif isinstance(val, dict) and any(k.startswith(name+".") for k in BOUNDS):
                visit(val, name+".")
            else:
                raise ValueError("Unknown parameter: " + name)
    visit(value)
    if out["initialWorkers"] + out["initialSoldiers"] > min(out["cap"], out["tileCap"]):
        raise ValueError("Starting units must fit both population and tile limits.")
    return out


def rules(g):
    return dict(DEFAULTS, **g["parameters"], upkeepPeriod=10)


def js_string(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def seed_hash(*parts):
    h = 2166136261
    for char in "|".join("" if value is None else js_string(value) for value in parts):
        code = ord(char)
        if code > 0xffff:  # Match the existing engine's first UTF-16 unit per code point.
            code = 0xd800 + ((code - 0x10000) >> 10)
        h = ((h ^ code) * 16777619) & 0xffffffff
    h ^= h >> 16
    h = (h * 0x7feb352d) & 0xffffffff
    return (h ^ (h >> 15)) & 0xffffffff


def rng(seed):
    a = seed_hash(seed)
    def draw():
        nonlocal a
        a = (a + 0x6d2b79f5) & 0xffffffff
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296
    return draw


def distance(a, b):
    return abs(a % 8 - b % 8) + abs(a // 8 - b // 8)


def own(pos, side):
    return pos < 32 if side == 0 else pos >= 32


def neighbors(pos, side=0):
    return sorted((p for p in (pos-8,pos+1,pos+8,pos-1) if 0 <= p < 64 and distance(pos,p) == 1), reverse=bool(side))


def terrain(seed):
    draw = rng(seed)
    for _ in range(1000):
        tiles, pool = ["grass"] * 64, list(range(32))
        for i in range(31, 0, -1):
            j = int(draw() * (i+1))
            pool[i], pool[j] = pool[j], pool[i]
        n = 1 + int(draw()*3)
        for _ in range(n):
            pos, kind = pool.pop(), "mountain" if draw() < .5 else "water"
            tiles[pos] = tiles[63-pos] = kind
        for kind in ("wood","ore"):
            index = next(i for i in range(len(pool)-1,-1,-1) if pool[i] < 24)
            pos = pool.pop(index)
            tiles[pos] = tiles[63-pos] = kind
        start = next(i for i,t in enumerate(tiles) if t not in ("water","mountain"))
        seen, queue = {start}, [start]
        for pos in queue:
            for p in neighbors(pos):
                if p not in seen and tiles[p] not in ("water","mountain"):
                    seen.add(p)
                    queue.append(p)
        if len(seen) == 64 - n*2:
            return tiles
    raise ValueError("Could not generate connected terrain.")


def blocked(g, pos):
    return g["tiles"][pos] in ("water","mountain") or any(s["pos"] == pos and s["kind"] == "wall" for s in g["structures"])


def route(g, origin, target, side, through_walls=False):
    if origin == target:
        return []
    queue, previous = [origin], {origin:None}
    for current in queue:
        for pos in neighbors(current, side):
            if pos in previous or g["tiles"][pos] in ("water","mountain") or (not through_walls and any(s["pos"] == pos and s["kind"] == "wall" for s in g["structures"])):
                continue
            if any(p["side"] != side and p["base"] == pos for p in g["players"]) and pos != target:
                continue
            previous[pos] = current
            if pos == target:
                result = [pos]
                while current != origin:
                    result.append(current)
                    current = previous[current]
                return result[::-1]
            queue.append(pos)
    return None


def create(map_seed="1847", ai_seed="42", parameters=None):
    map_seed, ai_seed = js_string(map_seed), js_string(ai_seed)
    c = normalize_parameters(parameters)
    stats = dict.fromkeys(("wood","ore","food","trained","soldiersOrdered","soldiersLaunched","lost","damage","baseDamage","farms","walls","hungry","orders","moves","legalSum","legalMax","decisions"),0)
    return dict(version=RULES_VERSION, parameters=c, mapSeed=js_string(map_seed), aiSeed=js_string(ai_seed), turn=0, phase="setup", tiles=terrain(map_seed),
        players=[dict(side=s,base=None,hp=c["baseHP"],**c["initial"],hunger=0,upkeep=0,lastAttackedTurn=None,queue=None,nextId=0,stats=stats.copy()) for s in (0,1)],
        units=[],structures=[],events=[],result=None,playerSeeds=[js_string(ai_seed)+":A",js_string(ai_seed)+":B"])


def base_choices(g, side):
    return [p for p,t in enumerate(g["tiles"]) if own(p,side) and t == "grass" and sum(q != p and own(q,side) and t2 == "grass" and distance(p,q) <= 2 for q,t2 in enumerate(g["tiles"])) >= 2]


def make_unit(g, side, kind):
    p = g["players"][side]
    unit = dict(id=str(side)+":"+str(p["nextId"]),side=side,kind=kind,pos=p["base"],hp=rules(g)[kind+"HP"],job=dict(type="idle"),cargo=None)
    p["nextId"] += 1
    return unit


def place_bases(g, positions):
    if g["phase"] != "setup" or not isinstance(positions,list) or len(positions) != 2:
        raise ValueError("Two bases must be placed during setup.")
    result = deepcopy(g)
    for side in (0,1):
        if type(positions[side]) is not int or positions[side] not in base_choices(g,side):
            raise ValueError("Invalid base location.")
        result["players"][side]["base"] = positions[side]
    for side in (0,1):
        for kind in ("worker","soldier"):
            for _ in range(g["parameters"]["initialWorkers" if kind == "worker" else "initialSoldiers"]):
                result["units"].append(make_unit(result,side,kind))
    result["phase"] = "play"
    return result


def ai_base(g, side, seed=None):
    seed = g["aiSeed"] if seed is None else seed
    resources = [next(p for p,t in enumerate(g["tiles"]) if t == k and own(p,side)) for k in ("wood","ore")]
    def score(pos):
        return sum(len(route(g,pos,p,side) or []) or 20 for p in resources) + abs(pos//8-3.5)*.65 + seed_hash(seed,"base",63-pos if side else pos)%100/100
    return min(base_choices(g,side), key=score)


def can_pay(player, cost):
    return all(player[k] >= v for k,v in cost.items())


def pay(player, cost):
    for key, value in cost.items():
        player[key] -= value


def farm_job(job):
    return bool(job and (job["type"] in ("harvest","renew") or job["type"] == "build" and job["kind"] == "farm"))


def farm_available(g, unit, target):
    return not any(v["id"] != unit["id"] and v["kind"] == "worker" and any(farm_job(j) and j["target"] == target for j in (v["job"],v.get("pendingJob"))) for v in g["units"])


def build_choices(g, side, kind):
    base = g["players"][side]["base"]
    return [q for q,t in enumerate(g["tiles"]) if own(q,side) and distance(base,q) <= 2 and t == "grass"
        and not any(p["base"] == q for p in g["players"]) and not any(s["pos"] == q for s in g["structures"])
        and not any(j and j.get("target") == q and j["type"] == "build" for u in g["units"] for j in (u["job"],u.get("pendingJob")))
        and (kind != "wall" or not any(u["pos"] == q for u in g["units"])) and route(g,base,q,side) is not None]


def move_choices(g, unit):
    def allowed(pos):
        return not blocked(g,pos) and not any(p["side"] != unit["side"] and p["base"] == pos for p in g["players"])
    if unit["kind"] != "soldier":
        return [p for p in neighbors(unit["pos"],unit["side"]) if allowed(p)]
    seen, queue = {unit["pos"]}, [unit["pos"]]
    for pos in queue:
        for p in neighbors(pos,unit["side"]):
            if p not in seen and allowed(p):
                seen.add(p)
                queue.append(p)
    return queue[1:]


def legal(g, side):
    if type(side) is not int or side not in (0,1) or g["phase"] != "play":
        return []
    c, player = rules(g), g["players"][side]
    actions = [dict(type="wait")]
    if not player["queue"] and sum(u["side"] == side for u in g["units"]) < c["cap"]:
        actions += [dict(type="train",kind=k) for k in ("worker","soldier") if can_pay(player,c[k+"Cost"])]
    for u in g["units"]:
        if u["side"] != side:
            continue
        actions.append(dict(type="hold",unit=u["id"]))
        actions += [dict(type="move",unit=u["id"],target=p) for p in move_choices(g,u)]
        if u["kind"] == "soldier":
            actions.append(dict(type="assault",unit=u["id"]))
            continue
        actions += [dict(type="gather",unit=u["id"],resource=r) for r in ("wood","ore")]
        for kind in ("farm","wall"):
            if can_pay(player,c[kind+"Cost"]):
                actions += [dict(type="build",unit=u["id"],kind=kind,target=p) for p in build_choices(g,side,kind)]
        for f in g["structures"]:
            if f["side"] != side or f["kind"] != "farm" or not farm_available(g,u,f["pos"]):
                continue
            if not f["spent"]:
                actions.append(dict(type="harvest",unit=u["id"],target=f["pos"]))
            elif can_pay(player,c["renewCost"]) and not any(j and j["type"] == "renew" and j["target"] == f["pos"] for v in g["units"] for j in (v["job"],v.get("pendingJob"))):
                actions.append(dict(type="renew",unit=u["id"],target=f["pos"]))
    return actions


def action_key(action):
    return compact([[k,action[k]] for k in sorted(action)])


def will_queue(g, action):
    if action["type"] not in ("gather","build","renew","harvest"):
        return False
    u = next(u for u in g["units"] if u["id"] == action["unit"])
    return u["kind"] == "worker" and bool(u["cargo"] or u["pos"] != g["players"][u["side"]]["base"] and u["job"]["type"] in ("gather","harvest"))


def apply_order(g, side, action):
    c, p, kind = rules(g), g["players"][side], action["type"]
    if kind == "wait":
        return
    p["stats"]["orders"] += 1
    if kind == "train":
        if action["kind"] == "soldier":
            p["stats"]["soldiersOrdered"] += 1
        pay(p,c[action["kind"]+"Cost"])
        p["queue"] = dict(kind=action["kind"],left=c[action["kind"]+"Time"])
        return
    u = next(u for u in g["units"] if u["id"] == action["unit"])
    if kind in ("hold","move"):
        u.pop("pendingJob",None)
        u["job"] = dict(type="idle") if kind == "hold" else dict(type="move",target=action["target"])
    if kind == "assault":
        if not u.get("raidCommitted"):
            p["stats"]["soldiersLaunched"] += 1
            u["raidCommitted"] = True
        u["job"] = dict(type="assault",target=g["players"][1-side]["base"])
    job = None
    if kind == "gather":
        job = dict(type="gather",resource=action["resource"],target=next(q for q,t in enumerate(g["tiles"]) if t == action["resource"] and own(q,side)),work=0)
    if kind in ("build","renew"):
        pay(p,c[action["kind"]+"Cost"] if kind == "build" else c["renewCost"])
        job = dict(type=kind,kind=action.get("kind","farm"),target=action["target"],work=0)
    if kind == "harvest":
        job = dict(type="harvest",target=action["target"])
    if job:
        if will_queue(g,action):
            u["pendingJob"] = job
            g["events"].append(dict(type="job-queued",side=side,unit=u["id"],pos=u["pos"],job=deepcopy(job)))
        else:
            u["job"] = job
            u.pop("pendingJob",None)


def plan_orders(g, side, commands=None):
    commands = [] if commands is None else [commands] if isinstance(commands,dict) else commands
    if not isinstance(commands,list) or len(commands) > rules(g)["commandsPerTurn"]:
        raise ValueError("Commands must be a list within the per-turn limit.")
    preview, used = deepcopy(g), set()
    for action in commands:
        if not isinstance(action,dict) or not action or action_key(action) not in {action_key(a) for a in legal(preview,side)} or action.get("unit") in used or action.get("type") == "wait" and len(commands) > 1:
            raise ValueError("Illegal order for side " + str(side))
        if action.get("unit"):
            used.add(action["unit"])
        apply_order(preview,side,action)
    available = [] if len(commands) >= rules(g)["commandsPerTurn"] or any(a["type"] == "wait" for a in commands) else [a for a in legal(preview,side) if a.get("unit") not in used]
    return preview, available


def activate_pending(g, u):
    job = u.pop("pendingJob",None)
    if not job:
        return
    farm = next((f for f in g["structures"] if f["kind"] == "farm" and f["side"] == u["side"] and f["pos"] == job.get("target")),None)
    available = (not farm_job(job) or farm_available(g,u,job["target"])) and (
        bool(farm and not farm["spent"]) if job["type"] == "harvest" else bool(farm and farm["spent"]) if job["type"] == "renew" else
        not any(s["pos"] == job["target"] for s in g["structures"]) if job["type"] == "build" else True)
    u["job"] = job if available else dict(type="idle")
    g["events"].append(dict(type="job-start" if available else "job-cancel",side=u["side"],unit=u["id"],pos=u["pos"],job=deepcopy(job)))


def step(state, orders):
    """Validate both private queues, then resolve a simultaneous immutable turn."""
    if state["phase"] != "play":
        return deepcopy(state)
    if not isinstance(orders,list) or len(orders) != 2:
        raise ValueError("Provide two sealed command lists.")
    c = rules(state)
    counts = [len(legal(state,s)) for s in (0,1)]
    for side in (0,1):
        plan_orders(state,side,orders[side])
    g = deepcopy(state)
    g["turn"] += 1
    g["events"] = []
    for side in (0,1):
        stats = g["players"][side]["stats"]
        stats["legalSum"] += counts[side]
        stats["legalMax"] = max(stats["legalMax"],counts[side])
        stats["decisions"] += 1
        commands = orders[side]
        for action in [commands] if isinstance(commands,dict) else commands or []:
            apply_order(g,side,action)
    for farm in g["structures"]:
        if farm["kind"] == "farm" and farm["grow"] > 0:
            farm["grow"] -= 1
    for u in g["units"]:
        if u["kind"] != "worker" or u["cargo"] or u.get("pendingJob") or u["job"]["type"] != "harvest":
            continue
        farm = next((f for f in g["structures"] if f["kind"] == "farm" and f["side"] == u["side"] and f["pos"] == u["job"]["target"]),None)
        if farm and farm["spent"] and can_pay(g["players"][u["side"]],c["renewCost"]) and farm_available(g,u,farm["pos"]):
            pay(g["players"][u["side"]],c["renewCost"])
            u["job"] = dict(type="renew",kind="farm",target=farm["pos"],work=0)
            g["events"].append(dict(type="auto-renew",side=u["side"],unit=u["id"],pos=farm["pos"]))
    old = {u["id"]:u["pos"] for u in g["units"]}
    intended = {}
    for u in g["units"]:
        job = u["job"]
        target = g["players"][u["side"]]["base"] if u["cargo"] else job.get("target")
        next_pos = u["pos"]
        waiting = not u["cargo"] and not u.get("pendingJob") and job["type"] == "harvest" and any(f["pos"] == job["target"] and f["kind"] == "farm" and f["spent"] for f in g["structures"])
        if not waiting and target is not None and target != u["pos"]:
            adjacent = any(v["side"] != u["side"] and distance(u["pos"],v["pos"]) <= 1 for v in g["units"])
            if not (u["kind"] == "soldier" and adjacent):
                path = route(g,u["pos"],target,u["side"],job["type"] == "assault")
                if path and not blocked(g,path[0]) and not any(p["side"] != u["side"] and p["base"] == path[0] for p in g["players"]):
                    next_pos = path[0]
        intended[u["id"]] = next_pos
    while True:
        rejected = []
        for pos in range(64):
            occupants = [u for u in g["units"] if intended[u["id"]] == pos]
            if len(occupants) > c["tileCap"]:
                incoming = sorted((u for u in occupants if old[u["id"]] != pos), key=lambda u:seed_hash(g["playerSeeds"][u["side"]],g["turn"],"traffic",u["id"].split(":")[1]),reverse=True)
                if len(incoming) < len(occupants) - c["tileCap"]:
                    raise ValueError("Capacity invariant violated.")
                rejected.extend(incoming[:len(occupants)-c["tileCap"]])
        if not rejected:
            break
        for u in rejected:
            intended[u["id"]] = old[u["id"]]
    for u in g["units"]:
        target = intended[u["id"]]
        if target != u["pos"]:
            g["events"].append(dict(type="move",unit=u["id"],side=u["side"],**{"from":u["pos"],"to":target}))
            g["players"][u["side"]]["stats"]["moves"] += 1
            u["pos"] = target
    for u in g["units"]:
        p, job = g["players"][u["side"]], u["job"]
        stayed = old[u["id"]] == u["pos"]
        if u["cargo"]:
            if u["pos"] == p["base"]:
                resource, amount = u["cargo"]["resource"], u["cargo"]["amount"]
                p[resource] += amount
                p["stats"][resource] += amount
                g["events"].append(dict(type="delivery",side=u["side"],unit=u["id"],pos=u["pos"],resource=resource,amount=amount))
                u["cargo"] = None
                activate_pending(g,u)
            continue
        if u.get("pendingJob") and u["pos"] == p["base"] and job["type"] == "move":
            activate_pending(g,u)
            continue
        if job["type"] == "move" and u["pos"] == job["target"]:
            u["job"] = dict(type="idle")
        if not stayed or u["pos"] != job.get("target"):
            continue
        if job["type"] == "gather":
            job["work"] += 1
            if job["work"] >= c[job["resource"]+"GatherTime"]:
                u["cargo"] = dict(resource=job["resource"],amount=c[job["resource"]+"Load"])
                job["work"] = 0
                g["events"].append(dict(type="collect",unit=u["id"],side=u["side"],pos=u["pos"],resource=job["resource"]))
        if job["type"] == "build":
            if job["kind"] == "wall" and any(v["id"] != u["id"] and v["pos"] == u["pos"] for v in g["units"]):
                continue
            job["work"] += 1
            if job["work"] >= c["buildTime"]:
                if any(s["pos"] == job["target"] for s in g["structures"]):
                    u["job"] = dict(type="idle")
                    continue
                if job["kind"] == "wall":
                    escape = next((q for q in neighbors(u["pos"],u["side"]) if not blocked(g,q) and not any(p2["side"] != u["side"] and p2["base"] == q for p2 in g["players"]) and sum(v["pos"] == q for v in g["units"]) < c["tileCap"]),None)
                    if escape is None:
                        continue
                    origin, u["pos"] = u["pos"], escape
                    g["events"].append(dict(type="move",unit=u["id"],side=u["side"],**{"from":origin,"to":escape}))
                    p["stats"]["moves"] += 1
                structure = dict(kind=job["kind"],side=u["side"],pos=job["target"],hp=c["wallHP"],grow=c["growTime"] if job["kind"] == "farm" else 0)
                if job["kind"] == "farm":
                    structure["spent"] = False
                g["structures"].append(structure)
                p["stats"]["farms" if job["kind"] == "farm" else "walls"] += 1
                g["events"].append(dict(type="build",kind=job["kind"],side=u["side"],pos=job["target"]))
                u["job"] = dict(type="harvest",target=job["target"]) if job["kind"] == "farm" else dict(type="idle")
        if job["type"] == "renew":
            farm = next((f for f in g["structures"] if f["kind"] == "farm" and f["side"] == u["side"] and f["pos"] == job["target"]),None)
            if not farm or not farm["spent"]:
                u["job"] = dict(type="idle")
                continue
            job["work"] += 1
            if job["work"] >= c["renewTime"]:
                farm["spent"], farm["grow"] = False, c["growTime"]
                u["job"] = dict(type="harvest",target=job["target"])
                g["events"].append(dict(type="renew",side=u["side"],pos=job["target"]))
        if job["type"] == "harvest":
            farm = next((f for f in g["structures"] if f["kind"] == "farm" and f["side"] == u["side"] and f["pos"] == job["target"]),None)
            if not farm or farm["spent"]:
                if u.get("pendingJob"):
                    u["job"] = dict(type="move",target=p["base"])
                elif not farm:
                    u["job"] = dict(type="idle")
                continue
            if farm["grow"] == 0:
                u["cargo"], farm["spent"] = dict(resource="food",amount=c["harvest"]), True
                g["events"].append(dict(type="collect",unit=u["id"],side=u["side"],pos=u["pos"],resource="food"))
    # Select every attack before applying any damage.
    hits = []
    def target_key(u, side, kind_first=False):
        return ((0 if u["kind"] == "soldier" else 1,) if kind_first else ()) + (u["hp"],-u["pos"] if side else u["pos"],int(u["id"].split(":")[1]))
    for p in g["players"]:
        enemies = [u for u in g["units"] if u["side"] != p["side"] and u["kind"] == "soldier" and distance(p["base"],u["pos"]) <= 1]
        if enemies:
            target = min(enemies,key=lambda u:target_key(u,p["side"]))
            hits.append(dict(type="attack",source="hearth",unit="base:"+str(p["side"]),side=p["side"],**{"from":p["base"],"to":target["pos"]},target=target["id"],targetType="unit",damage=c["baseDefense"]))
    for u in g["units"]:
        enemies = [v for v in g["units"] if v["side"] != u["side"] and distance(u["pos"],v["pos"]) <= 1]
        target, kind = min(enemies,key=lambda v:target_key(v,u["side"],True)) if enemies else None, "unit"
        if not target and u["kind"] == "soldier":
            enemy = g["players"][1-u["side"]]
            if distance(u["pos"],enemy["base"]) <= 1:
                target, kind = dict(id="base:"+str(enemy["side"]),pos=enemy["base"]), "base"
        if not target and u["kind"] == "soldier":
            walls = [s for s in g["structures"] if s["kind"] == "wall" and s["side"] != u["side"] and distance(u["pos"],s["pos"]) <= 1]
            target = min(walls,key=lambda s:-s["pos"] if u["side"] else s["pos"]) if walls else None
            kind = "wall"
        if target:
            hits.append(dict(type="attack",unit=u["id"],side=u["side"],**{"from":u["pos"],"to":target["pos"]},target=target.get("id","wall:"+str(target["pos"])),targetType=kind,damage=c["soldierDamage"] if u["kind"] == "soldier" else c["workerDamage"]))
    for hit in hits:
        target = g["players"][1-hit["side"]] if hit["targetType"] == "base" else next(u for u in g["units"] if u["id"] == hit["target"]) if hit["targetType"] == "unit" else next(s for s in g["structures"] if s["pos"] == hit["to"] and s["kind"] == "wall")
        if hit["damage"] > 0:
            g["players"][1-hit["side"]]["lastAttackedTurn"] = g["turn"]
        target["hp"] -= hit["damage"]
        g["players"][hit["side"]]["stats"]["damage"] += hit["damage"]
        if hit["targetType"] == "base":
            g["players"][hit["side"]]["stats"]["baseDamage"] += hit["damage"]
    g["events"].extend(hits)
    for u in g["units"]:
        if u["hp"] <= 0:
            g["players"][u["side"]]["stats"]["lost"] += 1
            g["events"].append(dict(type="death",side=u["side"],unit=u["id"],pos=u["pos"]))
    g["units"] = [u for u in g["units"] if u["hp"] > 0]
    g["structures"] = [s for s in g["structures"] if s["hp"] > 0]
    for p in g["players"]:
        queue = p["queue"]
        if queue:
            queue["left"] = max(0,queue["left"]-1)
            if queue["left"] == 0 and sum(u["pos"] == p["base"] for u in g["units"]) < c["tileCap"] and sum(u["side"] == p["side"] for u in g["units"]) < c["cap"]:
                unit = make_unit(g,p["side"],queue["kind"])
                g["units"].append(unit)
                p["stats"]["trained"] += 1
                p["queue"] = None
                g["events"].append(dict(type="spawn",side=p["side"],unit=unit["id"],pos=p["base"]))
        upkeep = p["upkeep"] + sum(c[u["kind"]+"Upkeep"] for u in g["units"] if u["side"] == p["side"])
        cost, p["upkeep"] = divmod(upkeep,10)
        if cost:
            if p["food"] >= cost:
                p["food"], p["hunger"] = p["food"]-cost, 0
            else:
                p["food"], p["hunger"] = 0, p["hunger"]+1
                p["stats"]["hungry"] += 1
                g["events"].append(dict(type="hunger",side=p["side"],amount=p["hunger"],pos=p["base"]))
    defeated = ["base destroyed" if p["hp"] <= 0 else "starvation" if p["hunger"] >= c["starvationLimit"] else None for p in g["players"]]
    if any(defeated):
        g["result"] = dict(winner=None if all(defeated) else 1 if defeated[0] else 0,reason="mutual defeat" if all(defeated) else next(x for x in defeated if x))
    elif g["turn"] >= c["maxTurns"]:
        g["result"] = dict(winner=None,reason="turn limit")
    if g["result"]:
        g["phase"] = "finished"
    return g


def food_budget(g, side, kind=None, count=1):
    c, p = rules(g), g["players"][side]
    rate = sum(c[u["kind"]+"Upkeep"] for u in g["units"] if u["side"] == side) + (c[p["queue"]["kind"]+"Upkeep"] if p["queue"] else 0)
    return (c[kind+"Cost"]["food"]*count if kind else 0) + math.ceil((p["upkeep"]+(rate+(c[kind+"Upkeep"]*count if kind else 0))*15)/10)


def raid_progress(launched):
    size = 1
    while size < 5 and launched >= size:
        launched -= size
        size += 1
    if size == 5:
        launched %= 5
    return dict(size=size,sent=launched,remaining=size-launched)


def ai_plan(g, side, policy):
    c, p = rules(g), g["players"][side]
    workers = [u for u in g["units"] if u["side"] == side and u["kind"] == "worker"]
    soldiers = [u for u in g["units"] if u["side"] == side and u["kind"] == "soldier"]
    mid, late = g["turn"]/c["maxTurns"] >= .5, g["turn"]/c["maxTurns"] >= .7
    attacked = p["lastAttackedTurn"] is not None
    recent = attacked and g["turn"]-p["lastAttackedTurn"] <= 15
    worker_target = 4 if policy == "economy" else 3
    soldier_target = min(c["cap"]-len(workers),c["cap"] if policy == "rush" else (c["cap"] if mid else 1)+int(attacked) if policy == "balanced" else (c["cap"] if late else 3 if mid else 0)+(2 if attacked else 0))
    rate = (len(workers)*c["workerUpkeep"]+len(soldiers)*c["soldierUpkeep"]+(c[p["queue"]["kind"]+"Upkeep"] if p["queue"] else 0))/10
    raid = raid_progress(p["stats"]["soldiersLaunched"])
    waiting = sum(not u.get("raidCommitted") and u["job"]["type"] != "assault" for u in soldiers)
    remaining = max(1,min(raid["size"],max(1,c["cap"]-len(workers)))-raid["sent"])
    burst = max(0,remaining-waiting-int(bool(p["queue"] and p["queue"]["kind"] == "soldier"))) if policy == "rush" else 1
    reserve, military = food_budget(g,side), food_budget(g,side,"soldier",burst)
    food_target = max(reserve+math.ceil(rate*5),military+(c["harvest"] if policy == "economy" else math.ceil(c["harvest"]/2)))
    farm_target = min(max(1,len(workers)-1),max(2 if policy == "economy" else 1,math.ceil((rate+c["soldierUpkeep"]/10)*(c["renewTime"]+c["growTime"]+5)/c["harvest"])))
    path = route(g,p["base"],g["players"][1-side]["base"],side,True)
    travel = max(0,(len(path) if path is not None else c["maxTurns"])-1)
    threatened = any(u["side"] != side and distance(u["pos"],p["base"]) <= 3 for u in g["units"])
    return dict(mid=mid,late=late,attacked=attacked,recent=recent,workerTarget=worker_target,soldierTarget=soldier_target,rate=rate,burst=burst,
        reserve=reserve,militaryFood=military,foodTarget=food_target,farmTarget=farm_target,
        canArrive=c["maxTurns"]-g["turn"] > c["soldierTime"]+(0 if threatened else travel),
        raidSize=raid["size"],raidRemaining=remaining,raidReady=raid["sent"] > 0 or waiting >= remaining or c["maxTurns"]-g["turn"] <= travel+remaining*c["soldierTime"]+remaining)


def waiting_for_wood(g, u):
    return u["kind"] == "worker" and not u["cargo"] and not u.get("pendingJob") and u["job"]["type"] == "harvest" and not can_pay(g["players"][u["side"]],rules(g)["renewCost"]) and any(f["kind"] == "farm" and f["side"] == u["side"] and f["pos"] == u["job"]["target"] and f["spent"] for f in g["structures"])


def recovery_order(g, side, available=None):
    if g["phase"] != "play":
        return None
    workers = [u for u in g["units"] if u["side"] == side and u["kind"] == "worker"]
    if not any(waiting_for_wood(g,u) for u in workers) or any((u["cargo"] or {}).get("resource") == "wood" or any(j and j["type"] == "gather" and j["resource"] == "wood" for j in (u["job"],u.get("pendingJob"))) for u in workers):
        return None
    forest = next((p for p,t in enumerate(g["tiles"]) if t == "wood" and own(p,side)),None)
    if forest is None or route(g,forest,g["players"][side]["base"],side) is None:
        return None
    available = legal(g,side) if available is None else available
    choices = []
    for u in workers:
        if u["cargo"] or u.get("pendingJob") or not (u["job"]["type"] == "idle" or waiting_for_wood(g,u)):
            continue
        path = route(g,u["pos"],forest,side)
        action = next((a for a in available if a["type"] == "gather" and a["resource"] == "wood" and a["unit"] == u["id"]),None)
        if path is not None and action:
            choices.append((u,path,action))
    return min(choices,key=lambda x:(-int(x[0]["job"]["type"] == "idle"),len(x[1]),int(x[0]["id"].split(":")[1])))[2] if choices else None


def heuristic_order(g, side, policy, seed, available):
    recovery = recovery_order(g,side,available)
    if recovery:
        return recovery
    c, p, plan = rules(g), g["players"][side], ai_plan(g,side,policy)
    us = [u for u in g["units"] if u["side"] == side]
    workers, soldiers = [u for u in us if u["kind"] == "worker"], [u for u in us if u["kind"] == "soldier"]
    farms = sum(s["side"] == side and s["kind"] == "farm" and not s["spent"] for s in g["structures"]) + sum(bool(j and (j["type"] == "renew" or j["type"] == "build" and j["kind"] == "farm")) for u in workers for j in (u["job"],u.get("pendingJob")))
    food_need, urgent = p["food"] < plan["foodTarget"], p["food"] < plan["reserve"] or p["hunger"] > 0
    targets = dict(wood=max(c["workerCost"]["wood"],c["soldierCost"]["wood"]*plan["burst"])+c["farmCost"]["wood"]+c["renewCost"]["wood"]*2,
                   ore=max(c["workerCost"]["ore"],c["soldierCost"]["ore"]*plan["burst"]))
    guards = sorted((u for u in soldiers if u["job"]["type"] != "assault"),key=lambda u:(distance(u["pos"],p["base"]),int(u["id"].split(":")[1])))
    guard = guards[0] if policy == "economy" and not plan["late"] and guards else None
    def relative_key(a):
        b = dict(a,unit=a["unit"].split(":")[1] if a.get("unit") else None,target=(63-a["target"] if side else a["target"]) if "target" in a else None)
        return action_key(b)
    def score(a):
        kind = a["type"]
        if kind == "wait":
            return 0
        if kind == "hold":
            return -100
        if kind == "move":
            u = next(u for u in us if u["id"] == a["unit"])
            if distance(u["pos"],a["target"]) != 1 or policy != "rush" or u["kind"] != "soldier" or u.get("raidCommitted") or u["job"]["type"] == "assault" or plan["raidReady"] or u["pos"] != p["base"]:
                return -100
            if sum(v["pos"] == p["base"] for v in g["units"]) < c["tileCap"]-1 or sum(v["pos"] == a["target"] for v in g["units"]) >= c["tileCap"]:
                return -100
            return 110-distance(a["target"],g["players"][1-side]["base"])
        if kind == "train":
            if p["hunger"] or p["food"] < food_budget(g,side,a["kind"]):
                return -100
            if a["kind"] == "worker":
                if c["maxTurns"]-g["turn"] <= c["workerTime"]+c["growTime"]+2:
                    return -100
                if len(workers) < 2:
                    return 120
                return 72 if not plan["mid"] and len(workers) < plan["workerTarget"] else -100
            if not plan["canArrive"] or policy != "rush" and len(soldiers) >= plan["soldierTarget"] or p["food"] < plan["militaryFood"]:
                return -100
            if policy == "rush":
                return 85
            if policy == "balanced":
                return (90 if plan["mid"] else 42)+(18 if plan["recent"] else 10 if plan["attacked"] else 0)
            return (105 if plan["late"] else 80 if plan["mid"] else 35)+(20 if plan["recent"] else 10 if plan["attacked"] else 0)
        u = next(u for u in us if u["id"] == a["unit"])
        idle, busy = u["job"]["type"] == "idle", bool(u.get("pendingJob") or u["cargo"] or u["job"]["type"] in ("build","renew","harvest"))
        if kind == "assault":
            if u["job"]["type"] == "assault" or policy == "rush" and not u.get("raidCommitted") and not plan["raidReady"] or policy == "economy" and not plan["late"] and (u is guard or not plan["mid"]):
                return -100
            return 150 if policy == "economy" and plan["late"] else 130 if policy == "balanced" and plan["mid"] else 100
        if busy:
            return -100
        if kind == "harvest":
            if any(j and j["type"] == "harvest" and j["target"] == a["target"] for v in workers for j in (v["job"],v.get("pendingJob"))):
                return -100
            return (210 if urgent else 115 if food_need else 35)-distance(u["pos"],a["target"])*2
        if kind in ("build","renew"):
            if a.get("kind") == "wall" or farms >= plan["farmTarget"] or not food_need:
                return -100
            return (190 if urgent else 100 if p["food"] < plan["militaryFood"] else 65)+(12 if kind == "renew" else 0)+(8 if idle else 0)-distance(u["pos"],a["target"])*2-distance(p["base"],a["target"])*2
        if kind == "gather":
            if u["job"]["type"] == "gather" and u["job"]["resource"] == a["resource"]:
                return -100
            resource, stock = a["resource"], p[a["resource"]]
            others = sum(v["id"] != u["id"] and (v.get("pendingJob") or v["job"])["type"] == "gather" and (v.get("pendingJob") or v["job"])["resource"] == resource for v in workers)
            farm_wood = c["renewCost"]["wood"] if any(s["side"] == side and s["kind"] == "farm" and s["spent"] for s in g["structures"]) else c["farmCost"]["wood"]
            needs_wood = resource == "wood" and food_need and farms < plan["farmTarget"] and p["wood"] < farm_wood
            if not needs_wood and u["job"]["type"] == "gather" and p[u["job"]["resource"]] < targets[u["job"]["resource"]]:
                return -100
            return ((185 if urgent else 110) if needs_wood else 65 if stock < targets[resource] else 15 if stock < targets[resource]+c[resource+"Load"] else -40)-others*55+(8 if idle else -12)
        return -100
    def ranking(a):
        key = relative_key(a)
        return (-(score(a)+(seed_hash(seed,g["turn"],key)%1000/1000)*2),key)
    return min(available,key=ranking)


def choose_orders(g, side, policy="balanced", seed=None, choose=None):
    seed, commands = g["aiSeed"] if seed is None else seed, []
    choose = heuristic_order if choose is None else choose
    while len(commands) < rules(g)["commandsPerTurn"]:
        preview, available = plan_orders(g,side,commands)
        action = choose(preview,side,policy,seed,available)
        if not action or action["type"] == "wait":
            break
        if action_key(action) not in {action_key(a) for a in available}:
            raise ValueError("Controller returned an unavailable command.")
        commands.append(action)
    return commands


def relative_action(action, side):
    if isinstance(action,list):
        return [relative_action(a,side) for a in action]
    a = deepcopy(action)
    if a.get("unit"):
        original, local = a["unit"].split(":")
        a["unit"] = ("0" if int(original) == side else "1")+":"+local
    if "target" in a and side:
        a["target"] = 63-a["target"]
    return a


def observe(g, side):
    def pos(value):
        return 63-value if side else value
    def job(value):
        return dict(value, target=pos(value["target"])) if "target" in value else deepcopy(value)
    players = []
    for s in (side,1-side):
        p = g["players"][s]
        players.append(dict(base=pos(p["base"]),hp=p["hp"],wood=p["wood"],ore=p["ore"],food=p["food"],hunger=p["hunger"],lastAttackedTurn=p["lastAttackedTurn"],upkeep=p["upkeep"],queue=deepcopy(p["queue"])))
    units = []
    for u in g["units"]:
        v = deepcopy(u)
        v.update(id=("0" if u["side"] == side else "1")+":"+u["id"].split(":")[1],side=0 if u["side"] == side else 1,pos=pos(u["pos"]),job=job(u["job"]))
        if u.get("pendingJob"):
            v["pendingJob"] = job(u["pendingJob"])
        units.append(v)
    return dict(turn=g["turn"],parameters=deepcopy(g["parameters"]),tiles=g["tiles"][::-1] if side else g["tiles"][:],players=players,units=units,
        structures=[dict(s,side=0 if s["side"] == side else 1,pos=pos(s["pos"])) for s in g["structures"]])


def action_id(action, units):
    kind = action["type"]
    if kind == "wait":
        return 0
    if kind == "train":
        return 1 if action["kind"] == "worker" else 2
    slot = next(i for i,u in enumerate(units) if u["id"] == action["unit"])
    if kind in ("hold","assault"):
        op = 0 if kind == "hold" else 1
    elif kind == "gather":
        op = 2 if action["resource"] == "wood" else 3
    else:
        op = {"move":4,"build":68 if action.get("kind") == "farm" else 132,"renew":68,"harvest":196}[kind]+action["target"]
    return 3+slot*260+op


def encode(g, side, available=None, command_index=0):
    o, tokens = observe(g,side), [g["turn"]]
    tokens.extend(["grass","wood","ore","water","mountain"].index(t) for t in o["tiles"])
    for pos in range(64):
        s = next((s for s in o["structures"] if s["pos"] == pos),None)
        tokens.extend([1+s["side"]*2+int(s["kind"] == "wall"),s["hp"],5 if s.get("spent") else s["grow"]] if s else [0,0,0])
    def index(value, choices):
        return choices.index(value) if value in choices else -1
    kinds, resources, types = [None,"worker","soldier","farm","wall"], [None,"wood","ore","food"], ["idle","move","assault","gather","build","harvest","renew"]
    for s,p in enumerate(o["players"]):
        queue = p["queue"] or {}
        original = g["players"][side if s == 0 else 1-side]
        tokens.extend([p["base"],p["hp"],p["wood"],p["ore"],p["food"],p["hunger"],index(queue.get("kind"),kinds),queue.get("left",0),p["upkeep"],-1 if p["lastAttackedTurn"] is None else g["turn"]-p["lastAttackedTurn"],original["stats"]["soldiersLaunched"]])
    groups = [sorted((u for u in o["units"] if u["side"] == s),key=lambda u:int(u["id"].split(":")[1])) for s in (0,1)]
    for group in groups:
        for i in range(8):
            if i >= len(group):
                tokens.extend([-1]*16)
                continue
            u = group[i]
            j, pending, cargo = u["job"],u.get("pendingJob") or {},u["cargo"] or {}
            tokens.extend([index(u["kind"],kinds),u["pos"],u["hp"],index(j["type"],types),j.get("target",64),index(j.get("resource"),resources),index(j.get("kind"),kinds),j.get("work",0),index(cargo.get("resource"),resources),cargo.get("amount",0),
                index(pending.get("type"),types),pending.get("target",64),index(pending.get("resource"),resources),index(pending.get("kind"),kinds),pending.get("work",0),int(bool(u.get("raidCommitted")))])
    tokens.append(command_index)
    tokens = [min(127,max(-1,v))+1 for v in tokens]+[129]
    actions = legal(g,side) if available is None else available
    entries = sorted([(action_id(relative_action(a,side),groups[0]),a) for a in actions],key=lambda pair:pair[0])
    if len(tokens) != CONTEXT or len({i for i,_ in entries}) != len(entries):
        raise ValueError("Observation or action encoding invariant failed.")
    return dict(tokens=tokens,legal=[i for i,_ in entries],actions=[a for _,a in entries])


def decision_sequence(g, commands):
    commands = commands if isinstance(commands,list) else [commands]
    return commands+[dict(type="wait")] if len(commands) < rules(g)["commandsPerTurn"] and not any(a["type"] == "wait" for a in commands) else commands


def winner_rows(replay):
    result = replay["final"]["result"]
    if not result or result["winner"] is None:
        return []
    side, rows = result["winner"], []
    for i,pair in enumerate(replay["orders"]):
        g, prefix = replay["frames"][i], []
        for action in decision_sequence(g,pair[side]):
            preview, available = plan_orders(g,side,prefix)
            e = encode(preview,side,available,len(prefix))
            selected = next(i for i,a in enumerate(e["actions"]) if action_key(a) == action_key(action))
            rows.append(dict(tokens=e["tokens"],legal=e["legal"],target=e["legal"][selected]))
            prefix.append(action)
    return rows


def import_replay(entry):
    if not isinstance(entry,dict) or not all(k in entry for k in ("config","initial","orders")):
        raise ValueError("A match needs config, initial and orders.")
    config, initial, orders = entry["config"],entry["initial"],entry["orders"]
    if not isinstance(config,dict) or not isinstance(initial,dict):
        raise ValueError("Recorded config and initial state must be objects.")
    if "entrants" in entry and (not isinstance(entry["entrants"],list) or any(type(i) is not int for i in entry["entrants"]) or sorted(entry["entrants"]) != [0,1]):
        raise ValueError("Invalid entrant mapping.")
    if initial.get("version") != RULES_VERSION or config.get("rulesVersion",RULES_VERSION) != RULES_VERSION:
        raise ValueError("Match rules version differs.")
    if not all(isinstance(config.get(k),str) for k in ("mapSeed","aiSeed")) or not isinstance(config.get("policies"),list) or len(config["policies"]) != 2 or any(not isinstance(p,str) or len(p)>100 for p in config["policies"]):
        raise ValueError("A match needs string seeds and two policy names.")
    if "controllers" in config and (not isinstance(config["controllers"],list) or len(config["controllers"]) != 2 or any(p not in ("ai","human") for p in config["controllers"])):
        raise ValueError("Invalid recorded controllers.")
    parameters = normalize_parameters(config.get("parameters",initial.get("parameters")))
    seeds = config.get("playerSeeds",initial.get("playerSeeds"))
    if not isinstance(seeds,list) or len(seeds) != 2 or any(not isinstance(s,str) for s in seeds):
        raise ValueError("Two recorded player seeds are required.")
    g = create(config["mapSeed"],config["aiSeed"],parameters)
    g["playerSeeds"] = seeds[:]
    g = place_bases(g,[p["base"] for p in initial["players"]])
    def canonical(x):
        return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)
    if canonical(g) != canonical(initial):
        raise ValueError("Initial state differs from its seeds, bases and settings.")
    if not isinstance(orders,list) or len(orders) > parameters["maxTurns"]:
        raise ValueError("Invalid number of recorded turns.")
    frames = [deepcopy(g)]
    for pair in orders:
        if g["result"]:
            raise ValueError("Orders continue after the match ended.")
        g = step(g,pair)
        frames.append(deepcopy(g))
    if "frames" in entry and canonical(entry["frames"]) != canonical(frames) or "final" in entry and canonical(entry["final"]) != canonical(g):
        raise ValueError("Recorded frames or final state disagree with orders.")
    return dict(version=1,config=dict(deepcopy(config),parameters=parameters,playerSeeds=seeds[:],rulesVersion=RULES_VERSION),initial=frames[0],orders=deepcopy(orders),frames=frames,final=g)


def import_matches(value):
    if isinstance(value,dict) and value.get("format") == "hearthline-matches" and value.get("version") == 1:
        entries = value.get("matches")
    elif isinstance(value,dict) and value.get("version") == 1 and "initial" in value:
        entries = [value]
    else:
        raise ValueError("Use a Hearthline match collection.")
    if not isinstance(entries,list) or not entries:
        raise ValueError("A collection must contain at least one match.")
    return [import_replay(entry) for entry in entries]


def export_collection(replay):
    config = deepcopy(replay["config"])
    return dict(format="hearthline-matches",version=1,
        meta=dict(seed="imported",count=1,policies=config["policies"],paired=False,parameters=deepcopy(config["parameters"]),rulesVersion=RULES_VERSION,elapsedMs=0,mixedPolicies=False),
        matches=[dict(config=config,initial=deepcopy(replay["initial"]),orders=deepcopy(replay["orders"]),entrants=[0,1])])


class EngineClient:
    """Same request API as extra/server.py, entirely in the current Python process."""
    def __init__(self):
        self.models = json.loads((Path(__file__).resolve().parent / "models.json").read_text(encoding="utf-8"))["models"]
        self.cache, self.record, self.closed = {}, None, False

    def load_models(self, envelope):
        if not isinstance(envelope,dict) or envelope.get("format") != 1 or not isinstance(envelope.get("models"),dict) or not 1 <= len(envelope["models"]) <= 32:
            raise ValueError("Use a format-1 model envelope with 1-32 entries.")
        pending = []
        reserved = {"constructor","prototype","toString","toLocaleString","valueOf","hasOwnProperty","isPrototypeOf","propertyIsEnumerable"}
        for name, entry in envelope["models"].items():
            if not isinstance(entry,dict) or not isinstance(entry.get("config"),dict):
                raise ValueError("Model entries need a config object.")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",name) or name in reserved:
                raise ValueError("Invalid model name.")
            config = entry.get("config",{})
            d, h = config.get("width"),config.get("heads")
            if entry.get("encodingVersion") != ENCODING_VERSION or entry.get("trainingRulesVersion") != RULES_VERSION:
                raise ValueError("Incompatible model rules or encoding.")
            if config.get("context") != CONTEXT or config.get("vocab") != VOCAB or config.get("actions") != ACTIONS or type(d) is not int or not 4 <= d <= 256 or type(h) is not int or h < 1 or d % h or (type(config.get("layers",1)) is not int or config.get("layers",1) != 1):
                raise ValueError("Unsupported model architecture.")
            if not isinstance(entry.get("sha256"),str) or not re.fullmatch(r"[a-f0-9]{64}",entry["sha256"]):
                raise ValueError("A checkpoint SHA-256 is required.")
            normalize_parameters(entry.get("trainingParameters") or {})
            shapes = dict(wte=[VOCAB,d],wpe=[CONTEXT,d],wq=[d,d],wk=[d,d],wv=[d,d],wo=[d,d],fc1=[4*d,d],fc2=[d,4*d],lm_head=[ACTIONS,d])
            if set(entry.get("weights",{})) != set(shapes):
                raise ValueError("Invalid weight tensors.")
            for key, shape in shapes.items():
                tensor = entry["weights"][key]
                if not isinstance(tensor,dict):
                    raise ValueError("Weight tensors must be objects.")
                if tensor.get("shape") != shape or any(type(n) is not int for n in tensor["shape"]):
                    raise ValueError("Invalid weight shape: " + key)
                try:
                    binary = base64.b64decode(tensor["data"],validate=True)
                except (ValueError,TypeError) as error:
                    raise ValueError("Invalid weight data: " + key) from error
                if len(binary) != shape[0]*shape[1]*4 or any(not math.isfinite(v) for (v,) in struct.iter_unpack("<f",binary)):
                    raise ValueError("Invalid or non-finite weights: " + key)
            for field in ("episodes","positions","decisions","sampledDecisions","uniqueDecisionsSeen","parentWinningEpisodes"):
                if entry.get(field) is not None and (type(entry[field]) is not int or not 0 <= entry[field] <= 9007199254740991):
                    raise ValueError("Invalid training count: " + field)
            benchmark = entry.get("benchmark")
            if benchmark is not None:
                if not isinstance(benchmark,list):
                    raise ValueError("Invalid benchmark metadata.")
                for row in benchmark:
                    if not isinstance(row,dict) or not isinstance(row.get("opponent"),str) or any(type(row.get(k)) is not int or row[k] < 0 for k in ("matches","wins","losses","draws")) or row["wins"]+row["losses"]+row["draws"] != row["matches"]:
                        raise ValueError("Invalid benchmark metadata.")
            alias, number = name, 1
            while alias in self.models or alias in ("balanced","rush","economy","external") or any(p[0] == alias for p in pending):
                if alias in self.models and self.models[alias] == entry:
                    break
                alias, number = name[:54]+"-import"+str(number), number+1
            pending.append((alias,deepcopy(entry)))
        for name, entry in pending:
            if name not in self.models:
                self.models[name] = entry
        return [name for name,_ in pending]

    def model_order(self, g, side, policy, seed, available):
        if policy in ("balanced","rush","economy"):
            return heuristic_order(g,side,policy,seed,available)
        if policy not in self.models:
            raise ValueError("Unknown policy: " + str(policy))
        recovery = recovery_order(g,side,available)
        if recovery:
            return recovery
        try:
            import numpy as np
            from learning import MicroGPT
        except ImportError as error:
            raise RuntimeError("NumPy is required for trained policies; install requirements.txt.") from error
        if policy not in self.cache:
            entry = self.models[policy]
            config = {k:entry["config"][k] for k in ("context","vocab","actions","width","heads","seed")}
            model = MicroGPT(**config)
            model.p = {key:np.frombuffer(base64.b64decode(tensor["data"]),dtype="<f4").copy().reshape(tensor["shape"]) for key,tensor in entry["weights"].items()}
            self.cache[policy] = model
        encoded = encode(g,side,available,getattr(self,"command_index",0))
        scores = self.cache[policy].forward(np.asarray([encoded["tokens"]],dtype=np.int64),np.asarray([encoded["legal"]],dtype=np.int64))[0]
        return encoded["actions"][int(scores.argmax())]

    def orders(self, g, side, policy, seed):
        prefix = []
        while len(prefix) < rules(g)["commandsPerTurn"]:
            preview, available = plan_orders(g,side,prefix)
            self.command_index = len(prefix)
            action = self.model_order(preview,side,policy,seed,available)
            if action["type"] == "wait":
                break
            prefix.append(action)
        return prefix

    def new_record(self, config=None, external=True):
        config = {} if config is None else config
        if not isinstance(config,dict):
            raise ValueError("Config must be an object.")
        if config.get("rulesVersion",RULES_VERSION) != RULES_VERSION or config.get("heuristicVersion",HEURISTIC_VERSION) != HEURISTIC_VERSION:
            raise ValueError("Replay rules or controller version differs.")
        policies = config.get("policies",["external","rush"] if external else ["balanced","balanced"])
        known = {"balanced","rush","economy"} | set(self.models) | ({"external"} if external else set())
        if not isinstance(policies,list) or len(policies) != 2 or any(p not in known for p in policies):
            raise ValueError("Choose two known policies" + (" or external." if external else "."))
        for name, digest in config.get("modelCheckpoints",{}).items():
            if self.models.get(name,{}).get("sha256") != digest:
                raise ValueError("Model checkpoint changed: " + name)
        g = create(config.get("mapSeed","example-map" if external else "1847"),config.get("aiSeed","example-ai" if external else "42"),config.get("parameters"))
        seeds = config.get("playerSeeds",config.get("seeds",g["playerSeeds"]))
        if not isinstance(seeds,list) or len(seeds) != 2 or any(not isinstance(s,str) for s in seeds):
            raise ValueError("Two string player seeds are required.")
        g["playerSeeds"] = seeds[:]
        g = place_bases(g,config.get("bases",[ai_base(g,s,seeds[s]) for s in (0,1)]))
        cfg = dict(mapSeed=g["mapSeed"],aiSeed=g["aiSeed"],playerSeeds=seeds[:],policies=policies[:],rulesVersion=RULES_VERSION,heuristicVersion=HEURISTIC_VERSION,
            parameters=deepcopy(g["parameters"]),modelCheckpoints={p:self.models[p]["sha256"] for p in dict.fromkeys(policies) if p in self.models})
        return dict(version=1,config=cfg,initial=deepcopy(g),frames=[deepcopy(g)],orders=[],final=g)

    def snapshot(self):
        if self.record is None:
            raise ValueError("Start a match first.")
        g = self.record["final"]
        return dict(state=deepcopy(g),result=deepcopy(g["result"]),turn=g["turn"])

    def request(self, op, **message):
        if self.closed:
            raise RuntimeError("Engine is closed.")
        if op == "info":
            policies = [dict(id=k,label=v) for k,v in (("balanced","Steward"),("rush","Raider"),("economy","Cultivator"))]
            policies.extend(dict(id=k,label=k) for k in self.models)
            return dict(rulesVersion=RULES_VERSION,heuristicVersion=HEURISTIC_VERSION,encodingVersion=ENCODING_VERSION,context=CONTEXT,vocab=VOCAB,actions=ACTIONS,parameters=deepcopy(DEFAULTS),policies=policies)
        if op == "load_models":
            return dict(ids=self.load_models(message.get("models")))
        if op == "export_models":
            ids = message.get("ids",list(self.models))
            if not isinstance(ids,list) or any(i not in self.models for i in ids):
                raise ValueError("Choose known trained model IDs.")
            return dict(models=dict(format=1,models={i:deepcopy(self.models[i]) for i in ids}))
        if op == "simulate":
            replay = self.new_record(message.get("config"),False)
            while not replay["final"]["result"]:
                g = replay["final"]
                pair = [self.orders(g,s,replay["config"]["policies"][s],g["playerSeeds"][s]) for s in (0,1)]
                g = step(g,pair)
                replay["orders"].append(pair)
                replay["frames"].append(deepcopy(g))
                replay["final"] = g
            return dict(collection=export_collection(replay),result=deepcopy(g["result"]),turns=g["turn"])
        if op == "start":
            self.record = self.new_record(message.get("config"))
            return self.snapshot()
        if op == "state":
            return self.snapshot()
        if op == "export":
            self.snapshot()
            return dict(collection=export_collection(self.record))
        if op in ("legal","step"):
            self.snapshot()
            if self.record["final"]["result"]:
                raise ValueError("An unfinished match is required.")
            g = self.record["final"]
            if op == "legal":
                side, prefix = message.get("side"), message.get("prefix",[])
                if type(side) is not int or side not in (0,1):
                    raise ValueError("Side must be 0 or 1.")
                if not isinstance(prefix,list):
                    raise ValueError("Prefix must be a command list.")
                preview, available = plan_orders(g,side,prefix)
                e = encode(preview,side,available,len(prefix))
                recovery = recovery_order(preview,side,available)
                index = next((i for i,a in enumerate(e["actions"]) if recovery and action_key(a) == action_key(recovery)),None)
                return dict(e,observation=observe(preview,side),recoveryAction=e["legal"][index] if index is not None else None)
            orders = message.get("orders",[None,None])
            if not isinstance(orders,list) or len(orders) != 2:
                raise ValueError("Provide two sealed command lists.")
            pair = []
            for side, commands in enumerate(orders):
                if commands is None:
                    policy = self.record["config"]["policies"][side]
                    if policy == "external":
                        raise ValueError("External sides must provide a list (or [] to pass).")
                    commands = self.orders(g,side,policy,g["playerSeeds"][side])
                if not isinstance(commands,list):
                    raise ValueError("Commands must be lists.")
                pair.append(commands)
            next_state = step(g,pair)
            self.record["orders"].append(deepcopy(pair))
            self.record["frames"].append(deepcopy(next_state))
            self.record["final"] = next_state
            return self.snapshot()
        if op == "training_data":
            rows, episodes, parameters = [], 0, None
            for replay in import_matches(message.get("collection")):
                result = replay["final"]["result"]
                if not result or result["winner"] is None:
                    continue
                if parameters is not None and parameters != replay["config"]["parameters"]:
                    raise ValueError("Training matches must share one parameter snapshot.")
                parameters = replay["config"]["parameters"]
                episodes += 1
                rows.extend(winner_rows(replay))
            return dict(rows=rows,episodes=episodes,parameters=parameters,rulesVersion=RULES_VERSION,heuristicVersion=HEURISTIC_VERSION,encodingVersion=ENCODING_VERSION)
        raise ValueError("Unknown operation. Use info, load_models, export_models, simulate, start, state, legal, step, export or training_data.")

    def exchange(self, message):
        request_id = message.get("id") if isinstance(message,dict) else None
        try:
            if not isinstance(message,dict) or not isinstance(message.get("op"),str):
                raise ValueError("Requests need an op string.")
            args = {k:v for k,v in message.items() if k not in ("id","op")}
            return dict(id=request_id,ok=True,data=self.request(message["op"],**args))
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError,AttributeError,StopIteration) as error:
            return dict(id=request_id,ok=False,error=str(error) or "Invalid request.")

    def close(self):
        self.cache.clear()
        self.record = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-model", metavar="ID", help="Export one bundled model instead of serving stdio")
    parser.add_argument("--output", default="model.json")
    args = parser.parse_args()
    if hasattr(sys.stdin,"reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    with EngineClient() as engine:
        if args.export_model:
            try:
                model = engine.request("export_models",ids=[args.export_model])["models"]
                Path(args.output).write_text(json.dumps(model,indent=2,allow_nan=False)+"\n",encoding="utf-8")
            except (ValueError,OSError) as error:
                parser.exit(1,str(error)+"\n")
            return
        for line in sys.stdin:
            try:
                message = json.loads(line)
                response = engine.exchange(message)
            except (ValueError,TypeError) as error:
                response = dict(id=None,ok=False,error=str(error))
            # ASCII JSON also preserves Unicode seeds on legacy terminal encodings.
            print(json.dumps(response,separators=(",",":"),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
