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
- Python 3.8 or later (recommend Python 3.10 to package conflicts)
- PyTorch (latest stable version)
- Huggingface
- fair-esm
- pandas

### Installation

Create and activate a conda environment and install the required packages:

```bash
conda create -p /env_path/BioM3_env python=3.10 # /env_path/ is the location that contains the conda env
conda activate /env_path/BioM3_env
git clone https://huggingface.co/niksapraljak1/BioM3 /path/ 
cd /path/BioM3 # /path/ is the location that contains the huggingface repo for BioM3
sh torch_requirements.sh # install torch software
pip install -r requirements.txt # install remaining packages
```

## Model Weights Installation

Before running models, change directory to `BioM3/weights` folder, follow instructions, and download pretrained weights for the desired BioM3 configuration:

```bash
cd /path/BioM3/weights
# after changing directory, follow instructions of README.md to install weights for each model component
```

Note: choose the desired BioM3 configuration/checkpoint, then install weights for each folder:
- `/path/BioM3/weights/PenCL`
- `/path/BioM3/weights/Facilitator` 
- `/path/BioM3/weights/ProteoScribe`

Each folder contains a `README.md` detailing the different model weight configurations. For benchmarking, the optimal configuration is:
- `BioM3_PenCL_epoch20.bin`
- `BioM3_Facilitator_epoch20.bin`
- `BioM3_ProteoScribe_epoch20.bin`


## Stage 1: PenCL Inference

### Overview

This stage demonstrates how to perform inference using the **BioM3 PenCL model** for aligning protein sequences and text descriptions. The model computes latent embeddings for the given inputs and calculates **dot product scores** (similarities) with normalization.

### Model Weights

Before running the model, ensure you have:
- Configuration file: `stage1_config.json`
- Pre-trained weights: `BioM3_PenCL_epoch20.bin`

### Running the Model

1. Change directory to BioM3 repo:
```bash
cd /path/BioM3 # /path/ where is the location to the cloned BioM3 repo
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
- **Protein Sequence:** <span style="color:red">MAKEDNIEMQGTVLETLPNTMFRVELENGHVVTAHISGKMRKNYIRILTGDKVTVELTPYDLSKGRIVFRSR</span>  
- **Text Description:** <span style="color:blue">PROTEIN NAME: Translation initiation factor IF-1. FUNCTION: One of the essential components for the initiation of protein synthesis. Binds in the vicinity of the A-site. Stabilizes the binding of IF-2 and IF-3 on the 30S subunit to which N-formylmethionyl-tRNA(fMet) subsequently binds. Helps modulate mRNA selection, yielding the 30S pre-initiation complex (PIC). Upon addition of the 50S ribosomal subunit, IF-1, IF-2 and IF-3 are released leaving the mature 70S translation initiation complex. SUBUNIT: Component of the 30S ribosomal translation pre-initiation complex which assembles on the 30S ribosome in the order IF-2 and IF-3, IF-1 and N-formylmethionyl-tRNA(fMet); mRNA recruitment can occur at any time during PIC assembly. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the IF-1 family. LINEAGE: The organism lineage is Bacteria, Pseudomonadota, Gammaproteobacteria, Enterobacterales, Enterobacteriaceae, Escherichia. FAMILY NAMES: Family names are Translation initiation factor 1A / IF-1.</span>

**Pair 2:**
- **Protein Sequence:** <span style="color:red">MVKMIVGLGNPGSKYEKTKHNIGFMAIDNIVKNLDVTFTDDKNFKAQIGSTFINHEKVYFVKPTTFMNNSGIAVKALLTYYNIDITDLIVIYDDLDMEVSKLRLRSKGSAGGHNGIKSIIAHIGTQEFNRIKVGIGRPLKGMTVINHVMGQFNTEDNIAISLTLDRVVNAVKFYLQENDFEKTMQKFNG</span>  
- **Text Description:** <span style="color:blue">PROTEIN NAME: Peptidyl-tRNA hydrolase. FUNCTION: The natural substrate for this enzyme may be peptidyl-tRNAs which drop off the ribosome during protein synthesis. CATALYTIC ACTIVITY: an N-acyl-L-alpha-aminoacyl-tRNA + H2O = a tRNA + an N-acyl-L-amino acid + H(+). SUBUNIT: Monomer. SUBCELLULAR LOCATION: Cytoplasm. SIMILARITY: Belongs to the PTH family. LINEAGE: The organism lineage is Bacteria, Bacillota, Bacilli, Lactobacillales, Streptococcaceae, Streptococcus. FAMILY NAMES: Family names are Peptidyl-tRNA hydrolase.</span>

**Pair 3:**
- **Protein Sequence:** <span style="color:red">MTDYPIKYRLIKTEKHTGARLGEIITPHGTFPTPMFMPVGTQATVKTQSPEELKAIGSGIILSNTYHLWLRPGDELIARSGGLHKFMNWDQPILTDSGGFQVYSLADSRNITEEGVTFKNHLNGSKMFLSPEKAISIQNNLGSDIMMSFDECPQFYQPYDYVKKSIERTSRWAERGLKAHRRPHDQGLFGIVQGAGFEDLRRQSAADLVAMDFPGYSIGGLAVGESHEEMNAVLDFTTPLLPENKPRYLMGVGAPDSLIDGVIRGVDMFDCVLPTRIARNGTCMTSEGRLVVKNAKFAEDFTPLDHDCDCYTCQNYSRAYIRHLLKADETFGIRLTSYHNLYFLVNLMKKVRQAIMDDNLLEFRQDFLERYGYNKSNRNF</span>  
- **Text Description:** <span style="color:blue">PROTEIN NAME: Queuine tRNA-ribosyltransferase. FUNCTION: Catalyzes the base-exchange of a guanine (G) residue with the queuine precursor 7-aminomethyl-7-deazaguanine (PreQ1) at position 34 (anticodon wobble position) in tRNAs with GU(N) anticodons (tRNA-Asp, -Asn, -His and -Tyr)...</span>

**Pair 4:**
- **Protein Sequence:** <span style="color:red">MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIVNEGLKAVAAGMNPMDLKRGIDKAVIAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLEDELDVVEGMQFDRGYLSPYFINKPDTGAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLVIIAEDVEGEALATLVVNTMRGIVKVAAVKAPGFGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEESAIQGRVAQIRKQIEEATSDYDREKLQERVAKLAGGVAVIKVGAATEVEMKEKKARVDDALHATRAAVEEGVVAGGGVALVRVAAKLAGLTGQNEDQNVGIKVALRAMEAPLRQIVSNAGEEPSVVANNVKAGDGNYGYNAATEEYGNMIDFGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKGDAPDLGAAGGMGGMGGMGGMM</span>  
- **Text Description:** <span style="color:blue">PROTEIN NAME: Chaperonin GroEL. FUNCTION: Together with its co-chaperonin GroES, plays an essential role in assisting protein folding. The GroEL-GroES system forms a nano-cage...</span>

**Pair 5:**
- **Protein Sequence:** <span style="color:red">MGKAIGIDLGTTNSVVAVVVGGEPVVIPNQEGQRTTPSVVAFTDKGERLVGQVAKRQAITNPENTIFSIKRLMGRKYNSQEVQEAKKRLPYKIVEAPNGDAHVEIMGKRYSPPEISAMILQKLKQAAEDYLGEPVTEAVITVPAYFDDSQRQATKDAGRIAGLNVLRIINEPTAAALAYGLDKKKEEKIAVYDLGGGTFDISILEIGEGVIEVKATNGDTYLGGDDFDIRVMDWLIEEFKKQEGIDLRKDRMALQRLKEAAERAKIELSSAMETEINLPFITADASGPKHLLMKLTRAKLEQLVDDLIQKSLEPCKKALSDAGLSQSQIDEVILVGGQTRTPKVQKVVQDFFGKEPHKGVNPDEVVAVGAAIQAAILKGEVKEVLLLDVTPLSLGIETLGGVFTKIIERNTTIPTKKSQIFTTAADNQTAVTIKVYQGEREMAADNKLLGVFELVGIPPAPRGIPQIEVTFDIDANGILHVSAKDLATGKEQSIRITASSGLSEEEIKKMIREAEAHAEEDRRKKQIAEARNEADNMIYTVEKTLRDMGDRISEDERKRIEEAIEKCRRIKDTSNDVNEIKAAVEELAKASHRVAEELYKKAGASQQGAGSTTQSKKEEDVIEAEVEDKDNK</span>  
- **Text Description:** <span style="color:blue">PROTEIN NAME: Chaperone protein DnaK. FUNCTION: Acts as a chaperone. INDUCTION: By stress conditions e.g. heat shock...</span>

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
Shape of z_p (protein latent): torch.Size([5, 512])
Shape of z_t (text latent): torch.Size([5, 512])

Magnitudes of z_p vectors: tensor([4.2894, 4.0314, 4.2747, 4.0478, 3.9959])
Magnitudes of z_t vectors: tensor([33.3649, 32.5055, 31.6935, 33.3630, 29.6486])

=== Dot Product Scores Matrix ===
tensor([[28.8613, -3.3248, -0.4564,  7.5766,  3.3064],
        [-0.7815, 28.2294, 10.3146,  3.9422, 11.2805],
        [-2.7591, 12.8974, 30.3760, -0.2481,  2.5218],
        [10.4455,  3.6447, -3.9202, 30.2053,  7.3378],
        [ 5.3883, 10.0869, -1.4182,  8.1128, 27.7488]])

=== Normalized Probabilities ===
Protein-Normalized Probabilities (Softmax across Proteins for each Text):
tensor([[1.0000e+00, 1.9778e-14, 4.0705e-14, 1.4876e-10, 2.4255e-11],
        [1.3374e-13, 1.0000e+00, 1.9384e-09, 3.9271e-12, 7.0454e-08],
        [1.8511e-14, 2.1949e-07, 1.0000e+00, 5.9466e-14, 1.1068e-11],
        [1.0049e-08, 2.1039e-11, 1.2746e-15, 1.0000e+00, 1.3665e-09],
        [6.3943e-11, 1.3208e-08, 1.5558e-14, 2.5430e-10, 1.0000e+00]])

Text-Normalized Probabilities (Softmax across Texts for each Protein):
tensor([[1.0000e+00, 1.0513e-14, 1.8512e-13, 5.7037e-10, 7.9733e-12],
        [2.5160e-13, 1.0000e+00, 1.6584e-08, 2.8327e-11, 4.3569e-08],
        [4.0702e-15, 2.5655e-08, 1.0000e+00, 5.0136e-14, 7.9997e-13],
        [2.6208e-09, 2.9167e-12, 1.5118e-15, 1.0000e+00, 1.1715e-10],
        [1.9452e-10, 2.1357e-08, 2.1524e-13, 2.9662e-09, 1.0000e+00]])

=== Homology Matrix (Dot Product of Normalized z_p) ===
tensor([[ 1.0000, -0.0706, -0.1477,  0.1752,  0.1810],
        [-0.0706,  1.0000,  0.1573,  0.0197,  0.2951],
        [-0.1477,  0.1573,  1.0000,  0.0767, -0.0990],
        [ 0.1752,  0.0197,  0.0767,  1.0000,  0.2231],
        [ 0.1810,  0.2951, -0.0990,  0.2231,  1.0000]])
```

## Stage 2: Facilitator Sampling

### Overview

In this stage, the **Facilitator model** takes the text embeddings (z_t) computed in Stage 1 and generates **facilitated embeddings (z_c)**. The facilitated embeddings align more closely with protein embeddings (z_p) and reduce discrepancies, as demonstrated by **Mean Squared Error (MSE)** and **Maximum Mean Discrepancy (MMD)** metrics.

### Model Weights

Before running the model, ensure you have:
- Configuration file: `stage2_facilitator_config.json`
- Pre-trained weights: `BioM3_Facilitator_epoch20.bin`

### Running the Facilitator Model

1. Run sampling:
```bash
python run_Facilitator_sample.py \
    --json_path "stage2_config.json" \
    --model_path "./weights/Facilitator/BioM3_Facilitator_epoch20.bin" \
    --input_data_path "test_PenCL_embeddings.pt" \
    --output_data_path "test_Facilitator_embeddings.pt"
```

Arguments:
- **json_path**: Path to the JSON configuration file
- **model_path**: Path to the pre-trained facilitator weights
- **input_data_path**: Path to the input embeddings (z_t and z_p) generated in Stage 1
- **output_data_path**: Path to save the facilitated embeddings (z_c)

### Expected Output

The script provides the following outputs:

1. **Latent Embedding Shapes**
   - z_t: Text embeddings
   - z_p: Protein embeddings
   - z_c: Facilitated embeddings

2. **Vector Magnitudes**
   - L2 norms of z_t, z_p, and z_c for a given batch

3. **Mean Squared Error (MSE)**
   - MSE between facilitated embeddings (z_c) and protein embeddings (z_p)
   - MSE between text embeddings (z_t) and protein embeddings (z_p)

4. **Maximum Mean Discrepancy (MMD)**
   - MMD between facilitated embeddings (z_c) and protein embeddings (z_p)
   - MMD between text embeddings (z_t) and protein embeddings (z_p)

### Sample Output

```plaintext
=== Facilitator Model Output ===
Shape of z_t (Text Embeddings): torch.Size([5, 512])
Shape of z_p (Protein Embeddings): torch.Size([5, 512])
Shape of z_c (Facilitated Embeddings): torch.Size([5, 512])

=== Norm (L2 Magnitude) Results for Batch Index 0 ===
Norm of z_t (Text Embedding): 33.364857
Norm of z_p (Protein Embedding): 4.289446
Norm of z_c (Facilitated Embedding): 3.976427

=== Mean Squared Error (MSE) Results ===
MSE between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): 0.013486
MSE between Text Embeddings (z_t) and Protein Embeddings (z_p): 1.937837

=== Max Mean Discrepancy (MMD) Results ===
MMD between Facilitated Embeddings (z_c) and Protein Embeddings (z_p): 0.000009
MMD between Text Embeddings (z_t) and Protein Embeddings (z_p): 0.004736
```

### What the Output Means

1. **Latent Shapes**:
   - Ensures that z_c has the same shape as z_p and z_t

2. **Norms**:
   - z_c is closer in magnitude to z_p compared to z_t, showing that the facilitator model effectively aligns the embeddings

3. **MSE**:
   - Lower MSE for z_c and z_p compared to z_t and z_p confirms that z_c approximates z_p better

4. **MMD**:
   - The MMD loss shows that the **distribution** of z_c is closer to z_p than the original z_t

### Saving the Output

The facilitated embeddings are saved to the specified output_data_path for further stages.


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

