# 011b: confirmation with long generations and the server's own counter

176 streams x 128 tokens, all completed, 21,549 tokens in 372 s:
- steady decode (sum of the streams' own rates): **70.2 tok/s**
- whole phase including the admission of 176 prompts (mean time to first token 50.5 s): **57.9 tok/s**
- server-side `/api/stats` token counter, 10 s samples while >= 170 requests were in flight:
  **median 70.5 tok/s, max 80.4** (mean 43.1: the first samples are the ramp, when requests are in
  flight but still prefilling).
So the fleet decodes at 64-70 tok/s once its streams are running; a run's overall figure depends
on how much of it is admission (37 tok/s for 32-token answers, 58 for 128-token answers, above 60
from about 200 tokens per answer).
