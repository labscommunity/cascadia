# Shim fixtures

- `loop_random.xml` / `.bin`: a 2-float model whose only RandomUniform sits inside a Loop body
  (top level: Parameter, Constant, Loop, Result). Built with the OpenVINO 2026.4.1 Python API
  (`opset13.loop` around `opset13.random_uniform`) and `ov.save_model`.
