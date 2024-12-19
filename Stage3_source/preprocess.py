

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Resize
import torchvision.transforms as T


#from numba import jit
import numpy as np
import pandas as pd


def get_mnist_dataset(args:any) -> DataLoader: 


    if args.dataset == 'normal':

        print(args.download)
        transform = Compose([ToTensor(), Resize(args.image_size), lambda x: x > 0.5])
        train_dataset = MNIST(root=args.data_root, download=True, transform=transform, train=True)
        train_dataloader = DataLoader(
                train_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True
        )

    elif args.dataset == 'sequence':

        transform = Compose([ToTensor(), Resize(args.image_size), lambda x: x > 0.5, T.Lambda(lambda x: torch.flatten(x).unsqueeze(0))])
        train_dataset = MNIST(root=args.data_root, download=True, transform=transform, train=True)
        train_dataloader = DataLoader(
                train_dataset,
                num_workers=args.workers,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True
        )

    else:
        print('Please picker either normal or sequence')
        quit()

    return train_dataloader




' Protein preprocessing tools '

#@jit(nopython=True)
def pad_ends(
        seqs: list,
        max_seq_length: int
    ) -> list:

    padded_seqs = [] # add padded gaps at the end of each sequence
    for seq in seqs:

        seq_length = len(seq)
        # number of padded tokens
        pad_need = max_seq_length - seq_length
        # add number of padded tokens to the end
        seq += '-'*pad_need

        padded_seqs.append(seq)

    return padded_seqs


# create numerical represented sqeuences
def create_num_seqs(seq_list: list) -> list:

    # tokenizer
    #tokens = ['*', '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>', '-']
    tokens = [ '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>', '-']
    # needed to lose these to the token list
    tokens = tokens + ['X', 'U', 'Z', 'B', 'O']
    token2int = {x:ii for ii, x in enumerate(tokens)}

    # empty list to hold num rep. seqs.
    num_seq_list = []
    for seq in seq_list:
        num_seq_list.append([token2int[aa] for aa in seq])

    return num_seq_list

# prepare the protein sequences
def prepare_protein_data(
        args: any,
        data_dict: dict
    ) -> (
            list,
            list
    ):
        
        print([key for key in data_dict.keys()])

        print('Prepare dataset')
        # prepare sequences
        seq_list = [seq.replace('-','') for seq in data_dict[args.sequence_keyname]]
        seq_list = [['<START>'] + list(seq) + ['<END>'] for seq in seq_list]
        seq_lens = [len(seq) for seq in seq_list]
    
        # Determine the maximum sequence length based on context window size
        max_seq_len = int(args.diffusion_steps)
            
        # Get indices of sequences that meet the criteria
        valid_indices = [i for i, seq in enumerate(seq_list) if len(seq) <= max_seq_len]

        # Filter num_seq_list based on these indices
        filter_seq_list = [seq_list[i] for i in valid_indices]
       
        max_seq_len = int(args.image_size * args.image_size)
        padded_seq_list = pad_ends(
                seqs=filter_seq_list,
                max_seq_length=max_seq_len
        )
        num_seq_list = create_num_seqs(padded_seq_list) # numerical representations

        # prepare class labels
        #class_label_list = df.label.values.tolist()
        if args.facilitator in ['MSE', 'MMD']:
            text_emb = data_dict['text_to_protein_embedding']
        elif args.facilitator in ['Default']:
            text_emb = data_dict['text_embedding']
        else:
            raise ValueError(f"Unexpected value for 'facilitator': {args.facilitator}")
        
        text_emb = [text_emb[i] for i in valid_indices] 
        # prune sequence and texts out based on length

        print('Finished preparing dataset')
        #  


        return (
                num_seq_list,
                text_emb
        )


class protein_dataset(Dataset):
    """

    Sequence dataloader

    """

    def __init__(
            self,
            num_seq_list: list,
            text_emb: torch.Tensor
    ):

        if not torch.is_tensor(num_seq_list):
            self.num_seqs = torch.tensor(num_seq_list).float()

        else:
            pass

        self.text_emb = text_emb

        #if not torch.is_tensor(class_label_list):
        #    self.class_label = torch.tensor(class_label_list).float()

    def __len__(self,):
        """
        number of samples total
        """
        return len(self.num_seqs)

    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor
    ):

        """
        extract adn return the data batch samples
        """

        # convert and return the data batch samples
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sequences
        num_seqs = self.num_seqs[idx]
        # class labels
        text_emb = self.text_emb[idx]

        return (
                num_seqs,
                text_emb
        )
