import yaml
from argparse import Namespace
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import Stage1_source.preprocess as prep
import Stage1_source.model as mod
import Stage1_source.PL_wrapper as PL_wrap


# Step 1: Load JSON configuration
def load_json_config(json_path):
    """
    Load JSON configuration file.
    """
    with open(json_path, "r") as f:
        config = json.load(f)
    # print("Loaded JSON config:", config)
    return config

# Step 2: Convert JSON dictionary to Namespace
def convert_to_namespace(config_dict):
    """
    Recursively convert a dictionary to an argparse Namespace.
    """
    for key, value in config_dict.items():
        if isinstance(value, dict):  # Recursively handle nested dictionaries
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)

def prepare_model(args) ->nn.Module:
    """
    Prepare the model and PyTorch Lightning Trainer using a flat args object.
    """
    model = mod.Facilitator(
            in_dim=args.emb_dim,
            hid_dim=args.hid_dim,
            out_dim=args.emb_dim,
            dropout=args.dropout
    )
    weights_path = f"{save_dir}/BioM3_Facilitator_epoch20.bin"# BioM3_PenCL_epoch20.bin" 
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully with weights!")
    return model

def compute_mmd_loss(x, y, kernel="rbf", sigma=1.0):
    """
    Compute the MMD loss between two sets of embeddings.
    Args:
        x: Tensor of shape [N, D]
        y: Tensor of shape [N, D]
        kernel: Kernel function, default is 'rbf' (Gaussian kernel)
        sigma: Bandwidth for the Gaussian kernel
    """
    def rbf_kernel(a, b, sigma):
        """
        Compute the RBF kernel between two tensors.
        """
        pairwise_distances = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-pairwise_distances / (2 * sigma ** 2))

    # Compute RBF kernel matrices
    K_xx = rbf_kernel(x, x, sigma)  # Kernel within x
    K_yy = rbf_kernel(y, y, sigma)  # Kernel within y
    K_xy = rbf_kernel(x, y, sigma)  # Kernel between x and y

    # Compute MMD loss
    mmd_loss = K_xx.mean() - 2 * K_xy.mean() + K_yy.mean()
    return mmd_loss


if __name__ == '__main__':
    
    json_path = f"{save_dir}/stage2_config.json"
    # Load and convert JSON config
    json_path = f"{save_dir}/stage2_config.json"
    config_dict = load_json_config(json_path)
    args = convert_to_namespace(config_dict) 

    # load model
    model =  prepare_model(args=args)
    
    # load test dataset
    embedding_dataset = torch.load('./PenCL_test_outputs.pt')

    # Run inference and store z_t, z_p

    with torch.no_grad():
        z_t = embedding_dataset['z_t']
        z_p = embedding_dataset['z_p'] 
        z_c = model(z_t)
        embedding_dataset['z_c'] = z_c

    # Compute MSE between embeddings
    mse_zc_zp = F.mse_loss(z_c, z_p)  # MSE between facilitated embeddings and protein embeddings
    mse_zt_zp = F.mse_loss(z_t, z_p)  # MSE between text embeddings and protein embeddings
 
    # Compute Norms (L2 magnitudes) for a given batch (e.g., first 5 embeddings)
    batch_idx = 0
    norm_z_t = torch.norm(z_t[batch_idx], p=2).item()
    norm_z_p = torch.norm(z_p[batch_idx], p=2).item()
    norm_z_c = torch.norm(z_c[batch_idx], p=2).item()
    
    # Compute MMD between embeddings
    MMD_zc_zp = model.compute_mmd(z_c, z_p)
    MMD_zp_zt = model.compute_mmd(z_p, z_t)

    # Print Results
    print("\n=== Facilitator Model Output ===")
    print(f"Shape of z_t (Text Embeddings): {z_t.shape}")
    print(f"Shape of z_p (Protein Embeddings): {z_p.shape}")
    print(f"Shape of z_c (Facilitated Embeddings): {z_c.shape}\n")

    print("=== Norm (L2 Magnitude) Results for Batch Index 0 ===")
    print(f"Norm of z_t (Text Embedding): {norm_z_t:.6f}")
    print(f"Norm of z_p (Protein Embedding): {norm_z_p:.6f}")
    print(f"Norm of z_c (Facilitated Embedding): {norm_z_c:.6f}")

    print("\n=== Mean Squared Error (MSE) Results ===")
    print(f"MSE between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): {mse_zc_zp:.6f}")
    print(f"MSE between Text Embeddings (z_t) and Protein Embeddings (z_p): {mse_zt_zp:.6f}")

    print("\n=== Max Mean Discrepancy (MMD) Results ===")
    print(f"MMD between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): {MMD_zc_zp:.6f}")
    print(f"MMD between Text Embeddings (z_t) and Protein Embeddings (z_p): {MMD_zp_zt:.6f}")

    print("\nFacilitator Model successfully computed facilitated embeddings!")

    # save output embeddings

    torch.save(embedding_dataset, 'Facilitator_test_outputs.pt')


