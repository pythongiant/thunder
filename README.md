START
  |
  v
[Experiment 1]
cuTile TurboQuant-K → QK
  |
  +── C < Triton ───────────────→ optimize tile shape / metadata layout
  |                                 |
  |                                 +── still < Triton → STOP cuTile path
  |
  +── C > Triton
        |
        v
[Experiment 2]
Fuse TurboQuant-V → PV
        |
        +── V fusion does not help
        |       |
        |       +── V decode-bound?
        |              |
        |              +── yes → optimize V unpack/reuse
        |              |
        |              +── no → keep existing V implementation
        |
        +── V fusion wins
                |
                v
[Experiment 3]
Full prefill kernel
Q → TurboQuant-K → QK
  → online softmax
  → TurboQuant-V → PV
                |
                +── slower than Triton/FA
                |       |
                |       +── identify:
                |            registers?
                |            shared memory?
                |            tensor-core utilization?
                |            unpack?
                |            memory?
                |       |
                |       +── fix one bottleneck at a time
                |
                +── faster
                        |
                        v
[Experiment 4]
GQA K reuse
                        |
                        +── no gain
                        |      → don't complicate kernel
                        |
                        +── gain
                               |
                               v
[Experiment 5]
Multi-head V reuse / GEMM formation
                               |
                               +── no gain
                               |      → keep simpler schedule
                               |
                               +── gain
                                      |
                                      v
[Experiment 6]
Decode kernel
split-K + TurboQuant K/V
                                      |
                                      +── loses at B=1
                                      |      → tune split count / tile
                                      |
                                      +── wins B=1
                                             |
                                             v
[Experiment 7]
High-batch decode
B=8,16,32
                                             |
                                             +── gain disappears
                                             |      → expected:
                                             |        memory advantage
                                             |        saturated
                                             |
                                             +── still wins
                                                    |
                                                    v
[Experiment 8]
Bitrate sweep
4/4
3/4
3/2
2/2
                                                    |
                                                    +── quality fails
                                                    |      → reject bitrate
                                                    |
                                                    +── quality passes
                                                           |
                                                           v
[Experiment 9]
Kernel specialization
model/head_dim/GQA/bitrate
                                                           |
                                                           v
[Experiment 10]
vLLM integration
                                                           |
                                                           v
[Experiment 11]
CUDA Graph / paged KV / real scheduler
                                                           |
                                                           v
[Experiment 12]
End-to-end benchmark
                                                           |
                                                           +── < target
                                                           |      → profile
                                                           |      → one last
                                                           |        kernel iteration
                                                           |
                                                           +── ≥ target
                                                                  |
                                                                  v
                                                               SHIP