# Context-Aware Sequential Model for Multi-Behaviour Recommendation

This repository contains the PyTorch implementation of the paper:
"Context-Aware Sequential Model for Multi-Behaviour Recommendation" by Shereen Elsayed, Ahmed Rashed, and Lars Schmidt-Thieme. https://arxiv.org/abs/2312.09684

## Abstract
Sequential recommendation models have recently become crucial for next-item recommendation tasks on various online platforms due to their ability to capture complex sequential patterns in historical user interactions. While many recent sequential models focus on modeling a single behavior (e.g., purchase), other implicit user interactions such as clicks and add-to-favorite can provide deeper insights into users' sequential behavior. This work proposes a Context-Aware Sequential Model (CASM) for multi-behavioral recommendations that leverages the advantages of sequential models and supports an arbitrary number of behaviors. Experimental results on four real-world datasets show that the proposed model significantly outperforms multiple state-of-the-art approaches.

<img width="868" alt="Screenshot 2024-05-30 at 11 51 04 AM" src="https://github.com/ariaattar/CASM-PyTorch/assets/72599441/c6eaef8b-c623-45a8-b170-602dd57dbac0">

## Performance Boost
Our CASM model shows a performance improvement of up to 19.24% over [CARCA](https://arxiv.org/abs/2204.06519) on various datasets.

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


## Requirements
- Python
- PyTorch 
- NumPy
- pandas

To install the required packages, run:

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/ariaattar/CASM-PyTorch.git
    cd CASM
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:
    ```bash
    python main.py
    ```


## Original TensorFlow Implementation
For the original TensorFlow implementation of the CASM model, please visit the following repository:
[CASM TensorFlow Implementation](https://github.com/Shereen-Elsayed/CASM)


## Citation
If you use this code for your research, please cite our paper:
```bibtex
@article{elsayed2023casm,
  title={Context-Aware Sequential Model for Multi-Behaviour Recommendation},
  author={Elsayed, Shereen and Rashed, Ahmed and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2312.09684},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
