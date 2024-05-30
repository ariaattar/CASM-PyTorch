# CASM-PyTorch
PyTorch Implementation of Context-Aware Sequential Model for Multi-Behaviour Recommendation https://arxiv.org/abs/2312.09684
<img width="884" alt="Screenshot 2024-05-30 at 1 42 28 AM" src="https://github.com/ariaattar/CASM-PyTorch/assets/72599441/ab0e4e4c-1e08-48a6-8dfe-449cdef26c4d">

| Method                           | MovieLens HR@10 | MovieLens NDCG@10 | Tianchi HR@10 | Tianchi NDCG@10 |
|----------------------------------|-----------------|--------------------|---------------|-----------------|
| **Sequential Recommendation Methods** |                 |                    |               |                 |
| SASRec                           | 0.911 ± 1𝐸−3    | 0.668 ± 5.1𝐸−3     | 0.659 ± 3𝐸−3  | 0.495 ± 2𝐸−3    |
| SSE-PT                           | 0.911 ± 7.1𝐸−3  | 0.657 ± 4.5𝐸−3     | 0.663 ± 1.2𝐸−2| 0.468 ± 1.3𝐸−2  |
| **Context-Aware Recommendation Methods** |                 |                    |               |                 |
| CARCA                            | 0.906 ± 2𝐸−3    | 0.665 ± 1𝐸−3       | 0.713 ± 4𝐸−4  | 0.500 ± 1𝐸−3    |
| **Multi-Behavior Recommendation Methods** |                 |                    |               |                 |
| MATN                             | 0.847           | 0.569              | 0.714 ± 7𝐸−4  | 0.485 ± 2𝐸−3    |
| KHGT                             | 0.861           | 0.597              | 0.652 ± 1𝐸−4  | 0.443 ± 1𝐸−4    |
| MB-STR                           | -               | -                  | -             | -               |
| MBHT                             | 0.913 ± 5.9𝐸−3  | 0.695 ± 7𝐸−3       | 0.725 ± 6.3𝐸−3| 0.554 ± 4.8𝐸−3  |
| **CASM**                         | **0.930 ± 6𝐸−4**| **0.713 ± 1.3𝐸−3** | **0.755 ± 9𝐸−4**| **0.584 ± 2.7𝐸−3** |
| **Improv.(%)**                   | **1.86%**       | **2.44%**           | **2.99%**     | **5.95%**       |

