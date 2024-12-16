import argparse
import yaml
from argparse import Namespace
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import Stage1_source.preprocess as prep
import Stage1_source.model as mod
import Stage1_source.PL_wrapper as PL_wrap

# Step 1: Load JSON Configuration
def load_json_config(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config

# Step 2: Convert JSON dictionary to Namespace
def convert_to_namespace(config_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_to_namespace(value)
    return Namespace(**config_dict)

# Step 3: Load Pre-trained Model
def prepare_model(config_args, model_path) -> nn.Module:
    model = mod.pfam_PEN_CL(args=config_args)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Model loaded successfully with weights!")
    return model

# Step 4: Prepare Test Dataset
def load_test_dataset(config_args):
    test_dict = {
        'primary_Accession': ['A0A009IHW8', 'A0A023I7E1'],
        'protein_sequence': [
            "MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKL...",
            "MRFQVIVAAATITMITSYIPGVASQSTSDGDDLFVPVSNFDPKSIFPEIKHP..."
        ],
        '[final]text_caption': [
            "PROTEIN NAME: 2' cyclic ADP-D-ribose synthase AbTIR...",
            "PROTEIN NAME: Glucan endo-1,3-beta-D-glucosidase 1..."
        ],
        'pfam_label': ["['PF13676']", "['PF17652','PF03639']"]
    }
    test_df = pd.DataFrame(test_dict)
    test_dataset = prep.TextSeqPairing_Dataset(args=config_args, df=test_df)
    return test_dataset

# Step 5: Argument Parser Function
def parse_arguments():
    parser = argparse.ArgumentParser(description="BioM3 Inference Script (Stage 1)")
    parser.add_argument('--json_path', type=str, required=True,
                        help="Path to the JSON configuration file (stage1_config.json)")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the pre-trained model weights (pytorch_model.bin)")
    return parser.parse_args()

# Step 6: Compute Homology Probabilities
def compute_homology_matrix(z_p_tensor):
    """
    Compute the homology matrix as cosine similarities between protein latent vectors.
    """
    # Normalize z_p to unit vectors
    z_p_normalized = F.normalize(z_p_tensor, p=2, dim=1)  # L2 normalization

    # Compute cosine similarity matrix
    homology_matrix = torch.matmul(z_p_normalized, z_p_normalized.T)  # (num_samples x num_samples)

    return homology_matrix


# Main Execution
if __name__ == '__main__':
    # Parse arguments
    config_args_parser = parse_arguments()

    # Load configuration
    config_dict = load_json_config(config_args_parser.json_path)
    config_args = convert_to_namespace(config_dict)

    # Load model
    model = prepare_model(config_args=config_args, model_path=config_args_parser.model_path)

    # Load test dataset
    test_dataset = load_test_dataset(config_args)

    # Run inference and store z_t, z_p
    z_t_list = []
    z_p_list = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            batch = test_dataset[idx]
            x_t, x_p = batch
            outputs = model(x_t, x_p, compute_masked_logits=False) # Infer Joint-Embeddings 
            z_t = outputs['text_joint_latent']  # Text latent
            z_p = outputs['seq_joint_latent']   # Protein latent
            z_t_list.append(z_t)
            z_p_list.append(z_p)

    # Stack all latent vectors
    z_t_tensor = torch.vstack(z_t_list)  # Shape: (num_samples, latent_dim)
    z_p_tensor = torch.vstack(z_p_list)  # Shape: (num_samples, latent_dim)

    # Compute Dot Product scores
    dot_product_scores = torch.matmul(z_p_tensor, z_t_tensor.T)  # Dot product

    # Normalize scores into probabilities
    protein_given_text_probs = F.softmax(dot_product_scores, dim=0)  # Normalize across rows (proteins), for each text
    text_given_protein_probs = F.softmax(dot_product_scores, dim=1)  # Normalize across columns (texts), for each protein

    # Compute magnitudes (L2 norms) for z_t and z_p
    z_p_magnitude = torch.norm(z_p_tensor, dim=1)  # L2 norm for each protein latent vector
    z_t_magnitude = torch.norm(z_t_tensor, dim=1)  # L2 norm for each text latent vector
    
    # Compute homology probabilities
    homology_matrix = compute_homology_matrix(z_p_tensor)

    # Print results
    print("\n=== Inference Results ===")
    print(f"Shape of z_p (protein latent): {z_p_tensor.shape}")
    print(f"Shape of z_t (text latent): {z_t_tensor.shape}")
    print(f"\nMagnitudes of z_p vectors: {z_p_magnitude}")
    print(f"Magnitudes of z_t vectors: {z_t_magnitude}")

    print("\n=== Dot Product Scores Matrix ===")
    print(dot_product_scores)

    print("\n=== Normalized Probabilities ===")
    print("Protein-Normalized Probabilities (Softmax across Proteins for each Text):")
    print(protein_given_text_probs)

    print("\nText-Normalized Probabilities (Softmax across Texts for each Protein):")
    print(text_given_protein_probs)

    print("\n=== Homology Matrix (Dot Product of Normalized z_p) ===")
    print(homology_matrix)

