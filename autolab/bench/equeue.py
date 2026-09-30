#!/usr/bin/env python3
"""The autolab's experiment queue: every experiment, past, running and proposed, as one file each.

    equeue.py list [--status S[,S..]] [--target TEXT]     table, next-to-run first
    equeue.py next                                        the item to run next (status ready, lowest priority number)
    equeue.py show ID                                     one item
    equeue.py add --title T --by WHO [--target T] [--priority N] [--exact yes|no|n/a] [--needs TEXT]
                 [--status proposed|ready] [--body FILE|-]     new item; prints its id and path
    equeue.py set ID key=value [key=value ...]            e.g. status=running owner=... experiment=030_uniform_frames
    equeue.py render                                      rewrite autolab/QUEUE.md from the items
    equeue.py backfill                                    one item per experiments/NNN_* folder that has none yet

Why one file per item (`autolab/queue/items/<id>.md`): several agents and people add items at the same time, from
different branches; separate files never conflict, an append to one shared list does. The id is the file name; it
is a slug, not an experiment number: numbers (`experiments/NNN_*`) are given when an item RUNS, by whoever runs it,
and recorded in the item's `experiment:` field. Standard library only. `autolab/queue/README.md` has the rules.
"""
import argparse, datetime, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
ITEMS = os.path.join(LAB, "queue", "items")
INDEX = os.path.join(LAB, "QUEUE.md")
STATUSES = ["running", "ready", "proposed", "blocked", "done", "dropped"]      # display order
OUTCOMES = ["kept", "reverted", "negative", "measurement", "inconclusive", ""]
FIELDS = ["id", "title", "status", "outcome", "priority", "target", "exact", "needs", "proposed_by", "owner",
          "experiment", "created", "updated"]


def today():
    return datetime.date.today().isoformat()


def slug(text):
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60].rstrip("-") or "item"


def parse(path):
    text = open(path).read()
    meta, body = {}, text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            for line in text[4:end].split("\n"):
                if ":" in line and not line.lstrip().startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = text[end + 5:]
    meta.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    return meta, body


def dump(meta, body):
    keys = FIELDS + sorted(k for k in meta if k not in FIELDS)
    head = "\n".join("%s: %s" % (k, meta.get(k, "")) for k in keys if k in meta or k in ("status", "title"))
    return "---\n%s\n---\n%s" % (head, body if body.startswith("\n") or not body else "\n" + body)


def load_all():
    out = []
    if os.path.isdir(ITEMS):
        for f in sorted(os.listdir(ITEMS)):
            if f.endswith(".md"):
                m, b = parse(os.path.join(ITEMS, f))
                out.append((m, b, os.path.join(ITEMS, f)))
    return out


def order(item):
    m = item[0]
    st = m.get("status", "proposed")
    try:
        pr = float(m.get("priority") or 99)
    except ValueError:
        pr = 99
    exp = m.get("experiment", "")
    return (STATUSES.index(st) if st in STATUSES else len(STATUSES), pr if st not in ("done", "dropped") else 0,
            exp if st in ("done", "dropped") else "", m.get("created", ""), m["id"])


def find(ident):
    items = load_all()
    hits = [it for it in items if it[0]["id"] == ident]
    if not hits:   # a unique prefix of the id, or of the experiment folder, is enough
        hits = [it for it in items if it[0]["id"].startswith(ident) or it[0].get("experiment", "").startswith(ident)]
    if len(hits) != 1:
        sys.exit("%d items match %r%s" % (len(hits), ident, "".join("\n  " + it[0]["id"] for it in hits)))
    return hits[0]


def table(items, wide=False):
    rows = []
    for m, _, _ in sorted(items, key=order):
        st = m.get("status", "")
        if st == "done" and m.get("outcome"):
            st = "done: " + m["outcome"]
        rows.append((m["id"], st, m.get("priority", ""), m.get("experiment", ""), m.get("exact", ""), m.get("target", ""),
                     m.get("title", ""), m.get("owner", "") or m.get("proposed_by", "")))
    return rows


def cmd_list(a):
    items = load_all()
    if a.status:
        want = set(a.status.split(","))
        items = [it for it in items if it[0].get("status") in want]
    if a.target:
        items = [it for it in items if a.target.lower() in it[0].get("target", "").lower()]
    for r in table(items):
        print("%-44s %-18s %-3s %-30s %-5s %s" % (r[0][:44], r[1], r[2], r[3][:30], r[4], r[6][:110]))


def cmd_next(a):
    ready = sorted([it for it in load_all() if it[0].get("status") == "ready"], key=order)
    if not ready:
        sys.exit("nothing is ready")
    m, body, path = ready[0]
    print(path)
    print(dump(m, body))


def cmd_show(a):
    m, body, path = find(a.id)
    print(path)
    print(dump(m, body))


def cmd_add(a):
    os.makedirs(ITEMS, exist_ok=True)
    ident = slug(a.title)
    n = 1
    while os.path.exists(os.path.join(ITEMS, ident + ".md")):
        n += 1
        ident = "%s-%d" % (slug(a.title), n)
    body = ""
    if a.body:
        body = sys.stdin.read() if a.body == "-" else open(a.body).read()
    if not body.strip():
        body = "\n## Hypothesis\n\n## Method\n\n## Prediction\n\n## Kill\n\n## Result\n"
    meta = dict(id=ident, title=a.title, status=a.status, outcome="", priority="%g" % a.priority, target=a.target,
                exact=a.exact, needs=a.needs, proposed_by=a.by, owner="", experiment="", created=today(), updated=today())
    path = os.path.join(ITEMS, ident + ".md")
    with open(path, "x") as f:
        f.write(dump(meta, body))
    print(ident)
    print(path)


def cmd_set(a):
    m, body, path = find(a.id)
    for kv in a.pairs:
        if "=" not in kv:
            sys.exit("expected key=value, got %r" % kv)
        k, v = kv.split("=", 1)
        if k == "status" and v not in STATUSES:
            sys.exit("status must be one of %s" % ", ".join(STATUSES))
        if k == "outcome" and v not in OUTCOMES:
            sys.exit("outcome must be one of %s" % ", ".join(o for o in OUTCOMES if o))
        m[k] = v
    m["updated"] = today()
    open(path, "w").write(dump(m, body))
    print(path)


def cmd_render(a):
    items = load_all()
    out = ["# Experiment queue", "",
           "Generated by `bench/equeue.py render` from `queue/items/*.md`: edit the items, not this file. How to add one:",
           "`queue/README.md`. Counts: " + ", ".join("%d %s" % (sum(1 for m, _, _ in items if m.get("status") == s), s)
                                                     for s in STATUSES) + ".", ""]
    for group, title in (("running", "Running"), ("ready", "Ready to run (in this order)"), ("proposed", "Proposed"),
                         ("blocked", "Blocked"), ("done", "Done"), ("dropped", "Dropped")):
        sel = [it for it in items if it[0].get("status") == group]
        if not sel:
            continue
        out += ["## " + title, ""]
        if group in ("done", "dropped"):
            out += ["| experiment | %s | what | item |" % ("outcome" if group == "done" else "why"), "|---|---|---|---|"]
            for m, _, _ in sorted(sel, key=order):
                why = m.get("outcome", "") if group == "done" else m.get("needs", "").replace("|", "/")
                out.append("| %s | %s | %s | [%s](queue/items/%s.md) |" % (m.get("experiment", ""), why,
                                                                            m.get("title", "").replace("|", "/"), m["id"], m["id"]))
        else:
            out += ["| prio | what | target | exact | needs | by / owner | item |", "|---|---|---|---|---|---|---|"]
            for m, _, _ in sorted(sel, key=order):
                who = m.get("owner") or m.get("proposed_by", "")
                out.append("| %s | %s | %s | %s | %s | %s | [%s](queue/items/%s.md) |" % (
                    m.get("priority", ""), m.get("title", "").replace("|", "/"), m.get("target", ""), m.get("exact", ""),
                    m.get("needs", "").replace("|", "/"), who, m["id"], m["id"]))
        out.append("")
    open(INDEX, "w").write("\n".join(out))
    print(INDEX)


def cmd_backfill(a):
    os.makedirs(ITEMS, exist_ok=True)
    have = {m.get("experiment") for m, _, _ in load_all()}
    exps = os.path.join(LAB, "experiments")
    for d in sorted(os.listdir(exps)):
        if not re.match(r"\d{3}", d) or d in have or not os.path.isdir(os.path.join(exps, d)):
            continue
        head = ""
        for name in ("verdict.md", "hypothesis.md"):
            p = os.path.join(exps, d, name)
            if os.path.exists(p):
                head = open(p).readline().strip().lstrip("# ").strip()
                break
        verdict = os.path.exists(os.path.join(exps, d, "verdict.md"))
        low = head.lower()
        outcome = ("kept" if "keep" in low else "reverted" if "revert" in low or "rolled back" in low else
                   "measurement" if verdict else "")
        title = re.sub(r"^\d{3}[a-z]?\s*(verdict)?\s*:?\s*", "", head, flags=re.I) or d
        meta = dict(id=d.replace("_", "-"), title=title, status="done" if verdict else "dropped", outcome=outcome,
                    priority="", target="", exact="", needs="", proposed_by="autolab", owner="autolab", experiment=d,
                    created="2026-09-20", updated=today())
        body = "\nRecord: `experiments/%s/` (hypothesis.md, verdict.md, overrides, phases).\n" % d
        path = os.path.join(ITEMS, meta["id"] + ".md")
        if not os.path.exists(path):
            open(path, "w").write(dump(meta, body))
            print("added", meta["id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--status", default=""); s.add_argument("--target", default=""); s.set_defaults(fn=cmd_list)
    s = sub.add_parser("next"); s.set_defaults(fn=cmd_next)
    s = sub.add_parser("show"); s.add_argument("id"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("add"); s.add_argument("--title", required=True); s.add_argument("--by", required=True)
    s.add_argument("--target", default=""); s.add_argument("--priority", type=float, default=50)
    s.add_argument("--exact", default="", choices=["", "yes", "no", "n/a"]); s.add_argument("--needs", default="")
    s.add_argument("--status", default="proposed", choices=["proposed", "ready"]); s.add_argument("--body", default="")
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("set"); s.add_argument("id"); s.add_argument("pairs", nargs="+"); s.set_defaults(fn=cmd_set)
    s = sub.add_parser("render"); s.set_defaults(fn=cmd_render)
    s = sub.add_parser("backfill"); s.set_defaults(fn=cmd_backfill)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
