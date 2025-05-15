## Environment Setup

We use Conda to manage the Python environment. Set up your environment with:

```bash
conda create -n safety-spaces python=3.10
conda activate safety-spaces
pip install -r requirements.txt
```

## Experiments

### 1. Fine-Tuning

Generate baseline models with various training objectives:

```bash
bash scripts/finetune.sh
```

This creates fully useful, fully harmful, and contaminated model variants for our experiments.

### 2. Projection Analysis

Compute model projections across different SVD fraction values:

```bash
python scripts/projection.py
```

This generates a comprehensive CSV with utility and harmfulness metrics for each projection configuration.

### 3. Subspace Updating

Run:

```bash
bash exp-3-update_spaces/update_spaces.sh
```

### 4. Activation Space Analysis

Run:

```bash
python exp-4-activation_spaces/activation_space.py
```

## License

[MIT License](LICENSE)