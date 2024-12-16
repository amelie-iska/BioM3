---
license: apache-2.0
---

# BioM3: Biological Multi-Modal Model for Protein Design

## Citation

If you use this code, please cite:

```bibtex
Natural Language Prompts Guide the Design of Novel Functional Protein Sequences
bioRxiv 2024.11.11.622734
doi: https://doi.org/10.1101/2024.11.11.622734
```

[Read the paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.11.622734v1)

## Software Requirements

### Required Dependencies
- Python 3.8 or later
- PyTorch (latest stable version)
- PyTorch Lightning
- pandas
- pyyaml

### Installation

Create and activate a conda environment:
```bash
conda create -n BioM3_env python=3.8
conda activate BioM3_env
```

Install the required packages:
```bash
conda install pytorch pytorch-lightning pandas pyyaml -c pytorch -c conda-forge
```

## Stage 1: PenCL Inference

### Overview

This stage demonstrates how to perform inference using the **BioM3 PenCL model** for aligning protein sequences and text descriptions. The model computes latent embeddings for the given inputs and calculates **dot product scores** (similarities) with normalization.

### Model Weights

Before running the model, ensure you have:
- Configuration file: `stage1_config.json`
- Pre-trained weights: `BioM3_PenCL_epoch20.bin`

### Running the Model

1. Clone the repository:
```bash
git clone https://huggingface.co/your_username/BioM3_PenCL
cd BioM3_PenCL
```

2. Run inference:
```bash
python run_PenCL_inference.py \
    --json_path "stage1_config.json" \
    --model_path "./weights/PenCL/BioM3_PenCL_epoch20.bin" \
    --output_path "test_PenCL_embeddings.pt"
```

### Example Input Data

The script demonstrates inference using two protein-text pairs from the SwissProt dataset:

**Pair 1:**
- **Protein Sequence:** MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKL...
- **Text Description:** PROTEIN NAME: 2' cyclic ADP-D-ribose synthase AbTIR...

**Pair 2:**
- **Protein Sequence:** MRFQVIVAAATITMITSYIPGVASQSTSDGDDLFVPVSNFDPKSIFPEIKHP...
- **Text Description:** PROTEIN NAME: Glucan endo-1,3-beta-D-glucosidase 1...

These pairs demonstrate how the model aligns protein sequences with their corresponding functional descriptions. The model will compute embeddings for both the sequences and descriptions, then calculate their similarities using dot product scores.

### Expected Output

The script provides the following outputs:

1. **Latent Embedding Shapes**
   - `z_p`: Protein sequence embeddings
   - `z_t`: Text description embeddings

2. **Vector Magnitudes**
   - L2 norms of both embedding types

3. **Dot Product Scores**
   - Similarity matrix between embeddings

4. **Normalized Probabilities**
   - Protein-normalized (softmax over rows)
   - Text-normalized (softmax over columns)

#### Sample Output
```plaintext
=== Inference Results ===
Shape of z_p (protein latent): torch.Size([2, 512])
Shape of z_t (text latent): torch.Size([2, 512])

Magnitudes of z_p vectors: tensor([5.3376, 4.8237])
Magnitudes of z_t vectors: tensor([29.6971, 27.6714])

=== Dot Product Scores Matrix ===
tensor([[ 7.3152,  1.8080],
        [ 3.3922, 16.6157]])

=== Normalized Probabilities ===
Protein-Normalized Probabilities:
tensor([[9.8060e-01, 3.7078e-07],
        [1.9398e-02, 1.0000e+00]])

Text-Normalized Probabilities:
tensor([[9.9596e-01, 4.0412e-03],
        [1.8076e-06, 1.0000e+00]])

=== Homology Matrix (Dot Product of Normalized z_p) ===
tensor([[1.0000, 0.1840],
        [0.1840, 1.0000]])

```

## Stage 2: Facilitator Sampling

🚧 **Coming Soon** 🚧

This stage will contain scripts and models for the Facilitator Sampling process. Check back for:
- Configuration files
- Model weights
- Running instructions
- Output examples

## Stage 3: ProteoScribe

🚧 **Coming Soon** 🚧

This stage will contain scripts and models for the ProteoScribe process. Check back for:
- Configuration files
- Model weights
- Running instructions
- Output examples

## Support

For questions or issues:
- Open an issue in this repository
- Contact: [Your contact information]

---
Repository maintained by the BioM3 Team

