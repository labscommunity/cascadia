# Discoveries

1. **A pipeline's admission order can starve it.** Rounds start at group 0 and requests arrive a
   few per round, so "admit into the group whose turn it is" fills the first groups only
   (15/10/8/6/4/4/2/0/0/0/0 at 48 streams). Every rank then idles two thirds of the time.
2. **Rows do not share expert reads.** MoE cost is 33-34 ms per row per rank at 1 and at 7 rows
   per frame, although 2 of each row's 8 experts are the same shared experts.
3. **`EXPERT_CACHE_MIB` is per layer**: 8000 MiB x 6 layers is the resident copy of the rank's
   experts (anon memory), next to a partial second copy in the page cache.
4. **Prompts over 256 tokens take the fleet down** (one StreamOpen frame over MAX_STREAM_ROWS).
5. **The fleet is two kinds of box**: ranks 0-7 are held at 25-31 W, ranks 8-10 run at 60 W and
   are 1.4x faster per stage.
