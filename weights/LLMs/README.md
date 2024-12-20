
---
### **`weights/LLMs/README.md`**

# LLMs Pre-trained Weights for Compiling PenCL

This folder contains the pre-trained weights for the **ESM2** and **PubMedBERT** models to compile **PenCL** model (Stage 1 of BioM3). The PenCL model aligns protein sequences and text descriptions to compute joint latent embeddings.

## Downloading Pre-trained Weights

### ESM2 Model
To download the ESM2 (650M parameter) model weights:
```bash
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
```

### PubMedBERT Model
To download the PubMedBERT model weights:
```bash
git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

## Usage
Once available, the pre-trained weights can be loaded as follows:

[Your usage instructions here]

## File Structure
After downloading, your weights directory should contain:
```
weights/
└── LLMs/
    ├── esm2_t33_650M_UR50D.pt
    └── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
```

Note: The PubMedBERT download will create a directory containing the model weights and configuration files, while ESM2 downloads as a single file.


