# Campaign template

Copy this for each new campaign. Numbered NNN_short_name.yaml.

## YAML schema

```yaml
version: 1
name: NNN_short_name
hypothesis: "One-sentence falsifiable claim. What you expect to find and why."
question: q1                              # FK to research_plan.yaml
moonshot_id: A3                           # FK to MOONSHOTS.md (which candidate is this testing)
moonshot: true                            # true = M+ magnitude class, false = XS-S
prior_art_note: "Brief lit-search summary; full notes in prior_art.md next to this file"

runner:
  backend: ssh
  rank0_host: cascadia-matias-02
  rank1_host: cascadia-matias-03
  command: "powershell -NoProfile -File k26_3prompt_eval.ps1"
  setup:
    - autolab/bench/kill_workers.sh
    - <any per-campaign setup, e.g. scp a modified binary>
    - autolab/bench/start_workers.sh
  timeout_seconds: 5400

defaults:
  max_tokens: 8
  temperature: 0
  prompts: ["The capital of France is", "The largest ocean on Earth is the", "Two plus two equals"]
  expected_substrs: ["paris", "pacific", "four"]

grid: {}                                  # Empty if no sweep; otherwise dict of varied params

metrics:
  primary: tok_per_sec
  direction: maximize
  collect:
    - name: tok_per_sec
      source: stdout
      pattern: '"tok_per_sec":([\d.]+),"quality_pass":true'
      type: float
    - name: quality_pass_count
      source: stdout
      pattern: '"quality_pass":"(\d+)/3"'
      type: int

repeats: 3                                # Independent runs for variance
acceptance:
  min_quality_pass: 3                     # must be 3/3 for tok/s to count
  significance: "best+median, σ < 10% of mean"

stopping: { max_failures: 3 }
```

## Per-campaign directory (`experiments/NNN_*/`)

- `prior_art.md` — literature notes from WebSearch / WebFetch pass before this campaign
- `notes.md` — design decisions, what got built, what surprised you
- `bench.jsonl` — raw bench output (each line is one prompt result + one aggregate line)
- `result.md` — verdict (win/neutral/negative), magnitude class, delta vs leader, link to spinout PR if win

## Result classification

| Outcome | Trigger |
|---------|---------|
| `win`     | tok/s increased ≥5% over current leader AND quality 3/3 AND reproducible across ≥3 runs |
| `neutral` | tok/s within ±5% of leader OR quality 3/3 with no significant delta |
| `negative` | tok/s decreased >5% OR quality <3/3 OR didn't run despite good-faith attempt |
| `parked`  | Infrastructure blocker; not blocking other moonshots; revisit later |
