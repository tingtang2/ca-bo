# ca-bo
Computation-aware Bayesian optimization

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

Use of virtual environments is optional but recommended. For GPU support refer to [the PyTorch docs to ensure correct the CUDA version](https://pytorch.org/get-started/locally/).

Also be sure to specify/change the `save_dir` and `data_dir` configs in the `main.py`.