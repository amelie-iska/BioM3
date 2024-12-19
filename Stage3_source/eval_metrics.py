"""
description: 
    metrics to compute model performance
"""

import Bio
from Bio.Align import substitution_matrices
import numpy as np
import matplotlib.pyplot as plt
import torch
import re

import Stage3_source.animation_tools as ani_tools


' compute Blosum62 soft accuracy '
class blosum_soft_accuracy:

    def __init__(self, ):

        self.blosum62 = substitution_matrices.load("BLOSUM62")
        self.alphabet = self.blosum62.alphabet

    def blosum_acc(
            self,
            aa1: str,
            aa2: str
        ) -> np.single:

        row = self.blosum62.alphabet.index(aa1)
        col = self.blosum62.alphabet.index(aa2)
        substitution_scores = self.blosum62[row, :].values()

        # Apply the softmax function to the substitution scores to get a prob dist.
        probs = np.exp(substitution_scores)/np.sum(np.exp(substitution_scores))

        # compute the soft acc. as the dot product of the prob dist. with a one-hot encoding
        # of the amino acid ...
        correct_aa = aa2
        correct_index = self.alphabet.index(correct_aa)
        one_hot = np.zeros_like(probs)
        one_hot[correct_index] = 1

        # normalize acc.
        soft_acc = np.dot(probs, one_hot) / np.max(probs)

        return soft_acc
    
    def split_seq(self, seq: str) ->list:
      #  no_pads = seq.count("<PAD>")
      #  split_seq = ["<START>"] + list(seq.replace("<START>","").replace("<END>","").replace("<PAD>","")) + ["<END>"] + ["<PAD>"] * no_pads
        split_seq = re.split(r'(-|<START>|<END>|<PAD>|(?<=\w)(?=\w))', seq)
        #split_seq = re.findall(r'<START>|<END>|<PAD>|[A-Z]|-|\*', seq)

        # remove empty strings and whitespace-only elements 
        split_seq = [char for char in split_seq if char and char.strip()]
        return split_seq
        
        

    def compute_soft_accuracy(
            self,
            seq1_list: list,
            seq2_list: list
        ) -> float:

        # make sure batch size matches
        if len(seq1_list) == len(seq2_list):
            self.batch_size = len(seq1_list)

        else:
            print("Please make sequence batch size equivalent...")

        # make sure sequence length matches
        if len(seq1_list[0]) == len(seq2_list[0]):
            self.L = len(seq1_list[0])

        else:
            #print("Please make sequence length match...")
            pass

        avg_soft_acc_per_batch = 0
        # loop over the batch of sequence
        for seq1, seq2 in zip(seq1_list, seq2_list):
            
            # split sequence into individual tokens
            seq1 = self.split_seq(seq1)
            seq2 = self.split_seq(seq2)
            # set number of positions
            self.L = len(seq2)
            self.L_h = 0 
            self.L_s = 0 
            avg_soft_acc_per_seq = 0
            avg_hard_acc_per_seq = 0

            # loop over the amino acid positions
            for aa1, aa2 in zip(seq1, seq2):
            
                if (aa1 not in ['-', '<START>', '<END>', '<PAD>']) and (aa2 not in ['-', '<START>', '<END>', '<PAD>']):
                    self.L_s += 1
                    soft_acc = self.blosum_acc(aa1=aa1, aa2=aa2)
                    avg_soft_acc_per_seq += soft_acc
                else:
                    self.L_h += 1
                    acc = 1*(aa1==aa2)
                    avg_hard_acc_per_seq += acc
            
            # compute accuracy for soft positions
            try:
                avg_soft_acc_per_seq *= 1/self.L_s
            except ZeroDivisionError:
                #print("L_s cannot be zero. Setting avg_soft_acc_per_seq to zero.")
                avg_soft_acc_per_seq = 0

            # compute accuracy for hard positions
            try:
                avg_hard_acc_per_seq *= 1/self.L_h
            except ZeroDivisionError:
                #print("L_h cannot be zero. Setting avg_hard_acc_per_seq to zero.")
                avg_hard_acc_per_seq = 0
        
                
            # compute the average accuracy between soft and hard
            if self.L_s == 0:
                avg_soft_acc_per_batch += avg_hard_acc_per_seq
            elif self.L_h == 0:
                avg_soft_acc_per_batch += avg_soft_acc_per_seq
            else:
                avg_soft_acc_per_batch += (avg_soft_acc_per_seq + avg_hard_acc_per_seq)/2

        avg_soft_acc_per_batch *= 1/self.batch_size
        return avg_soft_acc_per_batch


def compute_ppl(probs: torch.Tensor) -> float:

    batch_size, sequence_length, class_labels = probs.shape

    # flatten batch and sequence dimensions into a single dimension 
    flattened_probs = probs.reshape(batch_size * sequence_length, class_labels)

    # calc. perplexity for each sequence independently
    ppl = []
    for i in range(batch_size * sequence_length):
        sequence_probs = flattened_probs[i]
        # compute ppl per seq
        sequence_ppl = torch.exp(-torch.sum(
            sequence_probs * torch.log(sequence_probs)
            )
        )
        ppl.append(sequence_ppl.item())

    ppl = torch.tensor(ppl).view(batch_size, sequence_length) # ppl per sequence in a given batch
    avg_ppl = ppl.mean().item() # average ppl per batch

    return avg_ppl

def batch_compute_ppl(probs_list: list) -> float:
    
    batch_prob = sum([
        compute_ppl(probs=probs.unsqueeze(0).permute(0,2,1)) for probs in probs_list
    ]) / len(probs_list)
    
    return batch_prob


def compute_hard_acc(
        seq1: str,
        seq2: str
    ) -> float:
    
    
    hard_acc = sum([aa1 == aa2 for (aa1 ,aa2) in zip(seq1, seq2) if aa2 != '<PAD>'])
    valid_length = len([aa2 for aa2 in seq2 if aa2 != '<PAD>'])
    if valid_length == 0:
        return 1.0
    
    hard_acc /= valid_length
    
    return hard_acc

#def compute_hard_acc(
#        seq1: str,
#        seq2: str
#    ) -> float:
#
#    hard_acc = sum([aa1 == aa2 for (aa1 ,aa2) in zip(seq1, seq2)])
#    hard_acc *= 1/len(seq2)
#   return hard_acc

def batch_hard_acc(seq1_list: list, seq2_list: list) -> float:
    
    hard_acc = sum([
        compute_hard_acc(seq1=seq1, seq2=seq2) for (seq1,seq2) in zip(seq1_list, seq2_list)
    ]) / len(seq2_list)
    
    return hard_acc


def time_split_on_seq(
    seq: torch.Tensor,
    sample_seq_path: torch.Tensor,
    idx: torch.Tensor
    ) -> (
        list,
        list,
        list
    ):
    
    
    if len(seq.shape) != 2:
        batch_size, class_labels, _ = seq.shape
       
        # collect list
        current_seq, prev_seq, fut_seq = [], [], [] 
        
        for ii in range(batch_size):
            current_stack_probs, prev_stack_probs, fut_stack_probs = [], [], []

            for jj in range(class_labels):

                # current probs
                current_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] == idx.cpu()[ii])
                    ]
                )
                
                # prev probs 
                prev_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] < idx.cpu()[ii])
                    ]
                )

                # future probs 
                fut_stack_probs.append(
                    seq[ii,jj][
                        (sample_seq_path.cpu()[ii] > idx.cpu()[ii])
                    ]
                )
                
            current_seq.append(torch.stack(current_stack_probs))
            prev_seq.append(torch.stack(prev_stack_probs))
            fut_seq.append(torch.stack(fut_stack_probs))
       
    else:
        # split the sequences based on time indices
        current_seq = [seq[ii][sample_seq_path[ii] == idx[ii]] for ii in range(seq.shape[0])]
        prev_seq = [seq[ii][sample_seq_path[ii] < idx[ii]] for ii in range(seq.shape[0])]
        fut_seq = [seq[ii][sample_seq_path[ii] > idx[ii]] for ii in range(seq.shape[0])]
        
    return (
        current_seq,
        prev_seq,
        fut_seq
    )

@torch.no_grad()
def compute_acc_given_time_pos(
	real_tokens: torch.Tensor,
	sample_seq: torch.Tensor,
	sample_path: torch.Tensor,
	idx: torch.Tensor
	) -> (
	float,
	float,
	float,
	float,
	float,
	float
	):
            
    # tokenizer
    tokens = ['-', '<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>','<PAD>']
    #tokens = ['<START>', 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','<END>','<PAD>']
    tokens = tokens + ['X', 'U', 'Z', 'B', 'O']


    # split real tokens based on time indices
    current_real_tokens, prev_real_tokens, fut_real_tokens = time_split_on_seq(
    	seq=real_tokens.cpu(),
    	sample_seq_path=sample_path.cpu(),
    	idx=idx.cpu()
    )

    # split sampled tokens based on time indices
    current_sample_tokens, prev_sample_tokens, fut_sample_tokens = time_split_on_seq(
    	seq=sample_seq.cpu(),
    	sample_seq_path=sample_path.cpu(),
    	idx=idx.cpu()
    )

    # convert real sequences to characters
    current_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in current_real_tokens]
    prev_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in prev_real_tokens]
    fut_real_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in fut_real_tokens]

    # convert sample sequences to characters
    current_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in current_sample_tokens]
    prev_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in prev_sample_tokens]
    fut_sample_chars = [ani_tools.convert_num_to_char(tokens,seq_tokens) for seq_tokens in fut_sample_tokens]



    # drop empty entries in list (happens if t=0 or t=256)
    # prev string sequences
    prev_sample_chars = [item for item in prev_sample_chars if item]
    prev_real_chars = [item for item in prev_real_chars if item]
    # fut string sequences
    fut_real_chars = [item for item in fut_real_chars if item]
    fut_sample_chars = [item for item in fut_sample_chars if item]

    # class object to copmute blosum62 soft acc.
    soft_acc_tool = blosum_soft_accuracy()

    # split real sequence
    prev_real_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in prev_real_chars
    ]
    fut_real_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in fut_real_chars
    ]

    # split sample sequence
    prev_sample_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in prev_sample_chars
    ]
    fut_sample_split_chars = [
        soft_acc_tool.split_seq(sample) for sample in fut_sample_chars
    ]

    # compute hard and soft accuracy
    ' soft accuracy: '
    # positions < t ( aa positions)
    #prev_batch_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #    seq1_list=prev_sample_chars,
    #    seq2_list=prev_real_chars
    #)

    # positions > t ( aa positions)
    #fut_batch_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #    seq1_list=fut_sample_chars,
    #    seq2_list=fut_real_chars
    #)

    # positions = t (aa positions)
    #current_soft_acc = soft_acc_tool.compute_soft_accuracy(
    #seq1_list=current_sample_chars,
    #seq2_list=current_real_chars
    #)

    prev_batch_soft_acc, fut_batch_soft_acc, current_soft_acc = 0, 0, 0

    ' hard accuracy: '
    # positions < t ( aa positions)
    prev_batch_hard_acc = batch_hard_acc(
        seq1_list=prev_sample_split_chars,
        seq2_list=prev_real_split_chars 
    )

    # positions > t ( aa positions)
    fut_batch_hard_acc = batch_hard_acc(
        seq1_list=fut_sample_split_chars,
        seq2_list=fut_real_split_chars 
    )

    # positions = t (aa positions)
    current_hard_acc = compute_hard_acc(
        seq1=current_sample_chars,
        seq2=current_real_chars
    )

    return (
	prev_batch_hard_acc,
	prev_batch_soft_acc,
	fut_batch_hard_acc,
	fut_batch_soft_acc,
	current_hard_acc,
	current_soft_acc
    )


@torch.no_grad()
def compute_ppl_given_time_pos(
        probs: torch.Tensor,
        sample_path: torch.Tensor,
        idx: torch.Tensor
    ) -> (
            float,
            float,
            float
    ):

        current_probs, prev_probs, fut_probs = time_split_on_seq(
                probs.cpu(),
                sample_seq_path=sample_path.cpu(),
                idx=idx.cpu()
        )

        # ppl at the current time position (aa_i = t)
        # current_ppl = compute_ppl(probs=torch.stack(current_probs).permute(0,2,1))
        current_ppl = batch_compute_ppl(probs_list=current_probs)
        # ppl at the prev and fut time positions (aa_i < t and aa_i > t)
        prev_ppl = batch_compute_ppl(probs_list=prev_probs)
        fut_ppl = batch_compute_ppl(probs_list=fut_probs)

        return (
                current_ppl,
                prev_ppl,
                fut_ppl
        )
