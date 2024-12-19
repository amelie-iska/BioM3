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
                'primary_Accession': ["P69222", "B5XIP6", "B5XJL3", "B5Y368", "B5YH59"],
                    'protein_sequence': ["MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR",
                        "MVKMIVGLGNPGSKYEKTKHNIGFMAIDNIVKNLDVTFTDDKNFKAQIGSTFINHEKVYFVKPTTFMNNSGIAVKALLTYYNIDITDLIVIYDDLDMEVSKLRLRSKGSAGGHNGIKSIIAHIGTQEFNRIKVGIGRPLKGMTVINHVMGQFNTEDNIAISLTLDRVVNAVKFYLQENDFEKTMQKFNG",
                        "MTDYPIKYRLIKTEKHTGARLGEIITPHGTFPTPMFMPVGTQATVKTQSPEELKAIGSGIILSNTYHLWLRPGDELIARSGGLHKFMNWDQPILTDSGGFQVYSLADSRNITEEGVTFKNHLNGSKMFLSPEKAISIQNNLGSDIMMSFDECPQFYQPYDYVKKSIERTSRWAERGLKAHRRPHDQGLFGIVQGAGFEDLRRQSAADLVAMDFPGYSIGGLAVGESHEEMNAVLDFTTPLLPENKPRYLMGVGAPDSLIDGVIRGVDMFDCVLPTRIARNGTCMTSEGRLVVKNAKFAEDFTPLDHDCDCYTCQNYSRAYIRHLLKADETFGIRLTSYHNLYFLVNLMKKVRQAIMDDNLLEFRQDFLERYGYNKSNRNF",
                        "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIVNEGLKAVAAGMNPMDLKRGIDKAVIAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLEDELDVVEGMQFDRGYLSPYFINKPDTGAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLVIIAEDVEGEALATLVVNTMRGIVKVAAVKAPGFGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEESAIQGRVAQIRKQIEEATSDYDREKLQERVAKLAGGVAVIKVGAATEVEMKEKKARVDDALHATRAAVEEGVVAGGGVALVRVAAKLAGLTGQNEDQNVGIKVALRAMEAPLRQIVSNAGEEPSVVANNVKAGDGNYGYNAATEEYGNMIDFGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKGDAPDLGAAGGMGGMGGMGGMM",
                        "MGKAIGIDLGTTNSVVAVVVGGEPVVIPNQEGQRTTPSVVAFTDKGERLVGQVAKRQAITNPENTIFSIKRLMGRKYNSQEVQEAKKRLPYKIVEAPNGDAHVEIMGKRYSPPEISAMILQKLKQAAEDYLGEPVTEAVITVPAYFDDSQRQATKDAGRIAGLNVLRIINEPTAAALAYGLDKKKEEKIAVYDLGGGTFDISILEIGEGVIEVKATNGDTYLGGDDFDIRVMDWLIEEFKKQEGIDLRKDRMALQRLKEAAERAKIELSSAMETEINLPFITADASGPKHLLMKLTRAKLEQLVDDLIQKSLEPCKKALSDAGLSQSQIDEVILVGGQTRTPKVQKVVQDFFGKEPHKGVNPDEVVAVGAAIQAAILKGEVKEVLLLDVTPLSLGIETLGGVFTKIIERNTTIPTKKSQIFTTAADNQTAVTIKVYQGEREMAADNKLLGVFELVGIPPAPRGIPQIEVTFDIDANGILHVSAKDLATGKEQSIRITASSGLSEEEIKKMIREAEAHAEEDRRKKQIAEARNEADNMIYTVEKTLRDMGDRISEDERKRIEEAIEKCRRIKDTSNDVNEIKAAVEELAKASHRVAEELYKKAGASQQGAGSTTQSKKEEDVIEAEVEDKDNK"],
                 '[final]text_caption': ["PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. Stabilizes the binding of IF-2 and IF-3 on the 30S subunit to which N-formylmethionyl-tRNA(fMet) subsequently binds. Helps modulate mRNA selection, yielding the 30S pre-initiation complex (PIC). Upon addition of the 50S ribosomal subunit, IF-1, IF-2 and IF-3 are released leaving the mature 70S translation initiation complex. SUBUNIT: Component of the 30S ribosomal translation pre-initiation complex which assembles on the 30S ribosome in the order IF-2 and IF-3, IF-1 and N-formylmethionyl-tRNA(fMet); mRNA recruitment can occur at any time during PIC assembly. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the IF-1 family. LINEAGE: The organism lineage is Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales, Enterobacteriaceae, Escherichia. FAMILY NAMES: Family names are Translation initiation factor 1A / IF-1.",
                     "PROTEIN NAME: Peptidyl-tRNA hydrolase. FUNCTION: The natural substrate for this enzyme may be peptidyl-tRNAs which drop off the ribosome during protein synthesis. CATALYTIC ACTIVITY: an N-acyl-L-alpha-aminoacyl-tRNA + H2O = a tRNA + an N-acyl-L-amino acid + H(+). SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the PTH family. LINEAGE: The organism lineage is Bacteria, Bacillota, Bacilli, Lactobacillales, Streptococcaceae, Streptococcus. FAMILY NAMES: Family names are Peptidyl-tRNA hydrolase.",
                     "PROTEIN NAME: Queuine tRNA-ribosyltransferase. FUNCTION: Catalyzes the base-exchange of a guanine (G) residue with the queuine precursor 7-aminomethyl-7-deazaguanine (PreQ1) at position 34 (anticodon wobble position) in tRNAs with GU(N) anticodons (tRNA-Asp, -Asn, -His and -Tyr). Catalysis occurs through a double-displacement mechanism. The nucleophile active site attacks the C1' of nucleotide 34 to detach the guanine base from the RNA, forming a covalent enzyme-RNA intermediate. The proton acceptor active site deprotonates the incoming PreQ1, allowing a nucleophilic attack on the C1' of the ribose to form the product. After dissociation, two additional enzymatic reactions on the tRNA convert PreQ1 to queuine (Q), resulting in the hypermodified nucleoside queuosine (7-(((4,5-cis-dihydroxy-2-cyclopenten-1-yl)amino)methyl)-7-deazaguanosine). CATALYTIC ACTIVITY: 7-aminomethyl-7-carbaguanine + guanosine(34) in tRNA = 7-aminomethyl-7-carbaguanosine(34) in tRNA + guanine. COFACTOR: Binds 1 zinc ion per subunit. PATHWAY: tRNA modification; tRNA-queuosine biosynthesis. SUBUNIT: Homodimer. Within each dimer, one monomer is responsible for RNA recognition and catalysis, while the other monomer binds to the replacement base PreQ1. SIMILARITY: Belongs to the queuine tRNA-ribosyltransferase family. LINEAGE: The organism lineage is Bacteria, Bacillota, Bacilli, Lactobacillales, Streptococcaceae, Streptococcus. FAMILY NAMES: Family names are Queuine tRNA-ribosyltransferase.",
                     "PROTEIN NAME: Chaperonin GroEL. FUNCTION: Together with its co-chaperonin GroES, plays an essential role in assisting protein folding. The GroEL-GroES system forms a nano-cage that allows encapsulation of the non-native substrate proteins and provides a physical environment optimized to promote and accelerate protein folding. CATALYTIC ACTIVITY: ATP + H2O + a folded polypeptide = ADP + phosphate + an unfolded polypeptide. SUBUNIT: Forms a cylinder of 14 subunits composed of two heptameric rings stacked back-to-back. Interacts with the co-chaperonin GroES. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the chaperonin (HSP60) family. LINEAGE: The organism lineage is Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales, Enterobacteriaceae, Klebsiella/Raoultella group, Klebsiella. FAMILY NAMES: Family names are TCP-1/cpn60 chaperonin family.",
                     "PROTEIN NAME: Chaperone protein DnaK. FUNCTION: Acts as a chaperone. INDUCTION: By stress conditions e.g. heat shock. SIMILARITY: Belongs to the heat shock protein 70 family. LINEAGE: The organism lineage is Bacteria, Nitrospirae, Thermodesulfovibrionia, Thermodesulfovibrionales, Thermodesulfovibrionaceae, Thermodesulfovibrio. FAMILY NAMES: Family names are Hsp70 protein."],
                 "pfam_label": ["['PF01176’]", "['PF01195’]", "['PF01702’]", "['PF00118’]", "['PF00012’]"]
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
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save output embeddings")

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
    text_list = []
    protein_list = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            batch = test_dataset[idx]
            x_t, x_p = batch
            outputs = model(x_t, x_p, compute_masked_logits=False) # Infer Joint-Embeddings 
            z_t = outputs['text_joint_latent']  # Text latent
            z_p = outputs['seq_joint_latent']   # Protein latent
            z_t_list.append(z_t)
            z_p_list.append(z_p)
            
            protein_sequence = test_dataset.protein_sequence_list[idx]
            text_prompt = test_dataset.text_captions_list[idx]
            text_list.append(text_prompt)
            protein_list.append(protein_sequence)


    # Stack all latent vectors
    z_t_tensor = torch.vstack(z_t_list)  # Shape: (num_samples, latent_dim)
    z_p_tensor = torch.vstack(z_p_list)  # Shape: (num_samples, latent_dim)
    
    # Prepare embedding dict.
    embedding_dict = {
            'sequence': protein_list,
            'text_prompts': text_list,
            'z_t': z_t_tensor,
            'z_p': z_p_tensor
    }

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
    
    torch.save(embedding_dict, config_args_parser.output_path)
