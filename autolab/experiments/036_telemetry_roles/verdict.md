# 036: telemetry identifies pipeline roles and places windows by receive time

Kept, harness only. The worker profile already identifies its pipeline role;
outer telemetry identifies the installed box. Analysis now preserves both
and reports role 0 from installed box 8 after the swap. Timing windows use
profile receive time translated through the enclosing poll's receive time,
so a clock a day ahead cannot put old windows into the current phase.
Repeated windows are counted once; diagnostic probe tags are excluded.

Validation: synthetic clock-skew / backlog / duplicate / role-swap tests,
legacy records without receive timestamps, and re-analysis of 032's raw
telemetry. Corrected role-0 round trips are 592 and 561 ms in its two
fifteen-stream phases; the old summary was reading installed box 0, a middle
stage without reply counters. No fleet change and no raw telemetry committed.
