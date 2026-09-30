# 002b pipelined speculation, n-gram drafts

Binary 6bad6fd8 (same as 002a), overrides = 002a + CASCADIA_STREAMS_SPEC=1.

Time per token = a*T + (1-a)*(D*T + ~T/2) with T ~ 51 ms, D = 11. The n-gram drafter only guesses
when the last 2-4 tokens occurred before (in the prompt or the output), so on short reasoning
answers most positions carry no guess at all. Expected: 1.0-1.3x on the three gate prompts
(few repeats), more on a prompt whose answer quotes its input. The stage profile reports
guesses sent / right / wrong: that acceptance figure is the result that matters, because it says
what a better drafter would be worth on this fleet.

Must hold: gate character-identical to the reference (speculation never chooses a token).
