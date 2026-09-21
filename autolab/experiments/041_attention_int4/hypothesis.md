# 041: price int4 attention before any serving rollout

Attention projections consume about 10 ms per six-layer frame at int8.
Halving stored weight bytes helps only if the int4 kernel reads them fast
enough: earlier dense FC measurements showed 26–31 GB/s at int4 versus
about 80 GB/s at int8. The candidate may be slower.

On role 5 only, before its worker loads, build temporary int8 and int4 IRs
from its existing layer-30 and layer-31 shells. Compile with f16 inference,
warm one/two-row shapes, report median QKVR/O times over 41 iterations and
output deviation from int8. No candidate IR is installed. A 600-second
timeout, once-only marker, signal-death check and 12-start cutoff bound the
canary. No library, power, serving weight, or updater changes.

Prediction: ordinary int4 FC is slower; if so, close without a numerical
serving rollout. A competitive result requires a concrete quality study
and the owner's numerical-policy decision before enabling it fleet-wide.
The Python microbenchmark passed on a tiny CPU fixture; a Docker wrapper
test verified rank selection, failure fallback, no retry, and start cutoff.
It measures a kernel choice, not end-to-end model quality or fleet speed.
