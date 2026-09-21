# The experiment queue

Every experiment of this autolab, finished, running or only proposed, is ONE file in `queue/items/`. The overview
(`../QUEUE.md`) is generated from them. Anyone may add an item: another agent, a teammate, a session working on
something else that notices a thing worth measuring. Only whoever operates the fleet runs them (one publisher at a
time: see `../README.md`).

## Add an item

    python3 autolab/bench/equeue.py add --title "what changes, in one line" --by "who proposes it" \
        --target "15 streams | single stream | first token | reliability | offline study" \
        --exact yes|no|n/a --needs "binary change | overrides only | offline, Mac Pro | ..." \
        --priority 50 --body idea.md          # or --body - to read the body from stdin
    python3 autolab/bench/equeue.py render     # refresh QUEUE.md, commit both

or write the file by hand (copy any item): the name is the id, a slug of the title. One file per item means two
branches that both add items never conflict. Do not number items: `experiments/NNN_*` numbers are given when an item
runs, by the operator, and recorded in the item's `experiment:` field.

The body is free Markdown with these headings; the more of them are filled in, the sooner the item can run:
**Hypothesis** (why it should help, with the arithmetic), **Method** (what changes: code, env, which ranks),
**Prediction** (a number), **Kill** (what result ends it), **Result** (filled in by the operator).

## Fields

| field | meaning |
|---|---|
| `status` | `proposed` (an idea) -> `ready` (specified well enough to run; the operator or the proposer sets it) -> `running` -> `done`; `blocked` (say on what in `needs`), `dropped` (say why in the body) |
| `outcome` | when done: `kept`, `reverted`, `negative` (measured, not worth keeping), `measurement` (no change intended), `inconclusive` |
| `priority` | lower runs first; 1-9 = next up, 10-49 = soon, 50 = default for new proposals |
| `target` | which goal it serves |
| `exact` | does the model's output stay what it is today? `no` needs the user's decision before a rollout |
| `needs` | what it takes: a new binary, overrides only, an infra file, an offline machine, a decision |
| `experiment` | `experiments/NNN_*` folder once it has run (hypothesis, verdict, overrides, phases live there) |

## Operating it

    equeue.py list --status running,ready     what is going on and what is next
    equeue.py next                            the item to run now
    equeue.py set <id> status=running owner="autolab session" experiment=030_uniform_frames
    equeue.py set <id> status=done outcome=kept
    equeue.py render

Rules that come from this fleet's history (see `../JOURNAL.md`): one change per release where the effect has to be
attributed; canaries on one rank, self-reverting; no traffic before the settle check says `steady 3/3`; raw
telemetry, lab host names, addresses and the proxy name never enter the repository.
