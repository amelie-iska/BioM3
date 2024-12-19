import itertools
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.distributions import Categorical

import Stage3_source.eval_metrics as eval_funcs

# functions adapted for token-based transformers instead of Unet images (hat tip to author: LukasMosser) 

' sample random paths '
def sample_random_path(
        batch_size: int,
        seq_length: int,
        device: str='device'
    ) -> torch.Tensor:

    # create a batch of random sampling paths
    random_paths = torch.stack(
            [torch.randperm(seq_length, device=device) for _ in range(batch_size)],
            axis=0
    )
    # sequential paths
    #random_paths = torch.stack(
    #        [torch.arange(seq_length, device=device) for _ in range(batch_size)],
    #        axis=0
    #)
    return random_paths

' create masks to indicate positions that we have sampled already '
def create_mask_at_random_path_index(
        sample_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:

    # create a mask that has 1s everywhere we've sampled and 0's everywhere else
    mask = (sample_random_path < idx)
    return mask

' create a (batched) mask of where we are now sampling '
def create_sampling_location_mask(
        sampled_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:

    # create a binary mask that has 1 at the current location for us to sample 
    sampling_location_mask = (sampled_random_path == idx).long()
    return sampling_location_mask

' create masks to indicate positions beyond the current sampling position '
def create_mask_at_future_path_index(
        sampled_random_path: torch.Tensor,
        idx: any,
        batch_size: int,
        seq_length: int
    ) -> torch.Tensor:

    # create a mask that has 1s everywhere were are not going to be sampling and 
    # 0's everywhere we previously and currently sampled
    sampling_future_mask = (sampled_random_path > idx).long()
    return sampling_future_mask

' sampling from the probability distribution '
def sample_from_conditional(conditional_prob: any) -> torch.Tensor:
    # sample from the categorical dist.
    return conditional_prob.sample().permute(0,2,1)

' compute entropy of the model predicted probability distribution '
def compute_entropy(conditional_prob: any) -> torch.Tensor:
    # we can directly compute the entropy of the categorical distribution
    return conditional_prob.entropy()

' sampling the time trajectory '
class exp_weight_time_sample:

    def __init__(self, timesteps: int, decay_rate: float):
        
        self.timesteps = timesteps
        self.decay_rate = decay_rate
        # compute the weight based on the exp function
        self.weights = torch.tensor(
                [torch.exp(-torch.tensor([i])*decay_rate) for i in range(self.timesteps)]
        )

        # normalize weights
        self.weights /= self.weights.sum()

    def sample(self, batch_size: int) -> torch.Tensor:
        # generate random samples
        samples = torch.multinomial(self.weights, batch_size, replacement=True)
        return samples

def sample_random_index_for_sampling(
        batch_size: int,
        seq_length: int,
        device: str='cuda',
        option: str='random'
    ) -> any:

    if option == 'random':
        # sample a random index where we want to sample next
        idx = torch.randint(
                low=0,
                high=seq_length+1,
                size=(batch_size,1),
                device=device, 
                requires_grad=False
        )

    elif option == 'weighted':
        time_sampler = exp_weight_time_sampler(timesteps=seq_length+1, decay_rate=0.005)
        # sample a weighted random index where we want to sample next
        idx = time_sampler.sample(batch_size=batch_size).unsqueeze(1).to(device)

    return idx

#' log probs from realization '
def log_prob_of_realization(
        args: any,
        conditional_prob: any,
        real_tokens: torch.Tensor
    ) -> torch.Tensor:
    # compute the log-prob of a given realization
    #log_prob = conditional_prob._categorical.log_prob(real_tokens.to(args.device))
    log_prob = conditional_prob._categorical.log_prob(real_tokens)
  #  log_prob = conditional_prob.log_prob(real_tokens.to(args.device))
    return log_prob


#' get the log probabilities of the unsampled locations '
#def log_prob_of_unsampled_locations(
#        log_prob: torch.Tensor,
#        token_mask: torch.Tensor,
#        real_tokens: torch.Tensor
#    ) -> torch.Tensor:
#
#    # unsampled token positions (i.e. absorbing states)
#    unsampled_mask = (token_mask == 0) * 1
#    # non-padded tokens
#    non_padded_mask = (real_tokens != 23) * 1
#    # final mask is absorbing states that do not belong to padded tokens
#    final_unsampled_mask = unsampled_mask & non_padded_mask
#    # compute the total log prob of the unsampled locations, taking sum over log-probs
#    log_prob_unsampled = ( final_unsampled_mask * log_prob)
#    # sum log probs at absorbing positions 
#    summed_log_prob_unsampled = log_prob_unsampled.sum(1)
#
#    return summed_log_prob_unsampled


' get the log probabilities of the unsampled locations '
def log_prob_of_unsampled_locations(
        log_prob: torch.Tensor,
        token_mask: torch.Tensor
        ) -> torch.Tensor:

    # copmute the total log prob of the unsampled locations, taking sum over log-probs
    log_prob_unsampled = ((token_mask == 0)*1 * log_prob)

    return log_prob_unsampled.sum(1)

' weight the unsampeld log probs '
def weight_log_prob(
        log_prob_unsampled: torch.Tensor,
        idx: any,
        seq_length
    ) -> torch.Tensor:
    # compute the average log-prob over the unsampled locations
    log_prob_weighted = 1/(seq_length - idx.squeeze(1) + 1) * log_prob_unsampled
    return log_prob_weighted

' get mean log prob over the batch '
def compute_average_loss_for_batch(log_prob_weighted: torch.Tensor) -> torch.Tensor:
    # copute a (negative) average over the batch elements to copmute an unbiased estimator of the loss
    loss = -log_prob_weighted.mean()
    return loss

' create the numerical tokenized input data for transformer '
def create_token_labels(args, realization) -> (
        torch.Tensor,
        int,
        int
    ):

    bs, channel, seq_length = realization.size()
    temp_real = realization.reshape(bs, channel, seq_length)*1
    
    if args.task == 'MNIST':
        real_tokens = (temp_real == 1)*2 + (temp_real == 0)*1 # numerical tokeni labels for mnist
    
    elif args.task == 'proteins':
        real_tokens = temp_real + 1
    # background --> label 1
    # foreground --> label 2
    # mask (absorbing state) --> label 0
    return (
            real_tokens.squeeze(1),
            bs,
            seq_length
    )

' mask the positions for predictions/denoising ' 
def mask_realizations(
        real_tokens: torch.Tensor,
        random_path_mask: torch.Tensor
    ) -> torch.Tensor:

    out_real_tokens = real_tokens.clone()
    # batch size
    bs = random_path_mask.shape[0]
    # convert random path to boolean
    bool_rand_path_mask = random_path_mask.to(dtype=torch.bool)
    # positional masks
    # mask the future sample positions
    future_mask_positions = ((~bool_rand_path_mask)*1).squeeze(1)
    
    for ii in range(bs):
        
        mask_positions = future_mask_positions[ii].nonzero().tolist()
        # insert mask tokens
        out_real_tokens[ii, mask_positions] = 0

    return out_real_tokens


' model prediction '
def predict_conditional_prob(
        model: nn.Module,
        real_token_masked: torch.Tensor,
        idx: any,
        args: any
    ) -> (
            any,
            torch.Tensor
    ):
        #logits = model(x=real_token_masked.to(args.device), t=idx.view(-1,))
        logits = model(x=real_token_masked, t=idx.view(-1,))
        probs = F.softmax(
                logits,
                dim=1
        )

        conditional_prob = OneHotCategorical(probs=probs.permute(0,2,1))
        
        return (
                conditional_prob,
                probs
        )


"""
Here, we compute the previous position tokens, current token position, and future token positions, where
past, current, and future are defined by the time trajectory.
"""

' sample from model '
@torch.no_grad()
def sample_from_conditional(conditional_prob: any) -> torch.Tensor:
    # draw a sample from the categorical dist.
    cond_prob_sample = conditional_prob.sample().permute(0,2,1)
    return cond_prob_sample

' compute the accuracy at the current sampling location '
@torch.no_grad()
def sample_recover(
        real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        current_path_mask: torch.Tensor
    ) -> float:

    # remove from gpu
    real_tokens.cpu()
    cond_prob_sample.cpu()
    current_path_mask.cpu()

    # current sampling index
    current_tensor_pos = torch.argmax((current_path_mask == 1)*1, dim=-1)

    # model predictions match the ground truth label at current sampling index
    match_preds = [(
        real_tokens[seq_idx, ii] == torch.argmax(cond_prob_sample, dim=1)[seq_idx, ii]
        ).item()*1 for seq_idx, ii in enumerate(current_tensor_pos.cpu().numpy())
    ]

    return sum(match_preds)/len(match_preds)


' compute the accuracy of previous conditionally sampled locations '
@torch.no_grad()
def compute_prev_token_acc(
        cond_real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        path_mask: torch.Tensor
    ) -> np.ndarray:
    
    # remove from gpu
    cond_real_tokens.cpu()
    cond_prob_sample.cpu()
    path_mask.cpu()

    # class labels of the sampled model prediction
    cond_sample_tokens = torch.argmax(cond_prob_sample, dim=1)
    matches = []
    for ii , sample_pos in enumerate(path_mask):

        temp_real_tokens = cond_real_tokens[ii, sample_pos.nonzero()].squeeze(1)
        temp_sample_tokens = cond_sample_tokens[ii, sample_pos.nonzero()].squeeze(1)
        matches.append(
                (temp_real_tokens == temp_sample_tokens).tolist()
        )
      
    acc = []
    for match in matches:

        try: 
            acc.append(sum(match*1)/len(match))

        except ZeroDivisionError:
            acc.append(0)

    return np.mean(acc)


 
' compute the accuracy of previous conditionally sampled locations '
@torch.no_grad()
def compute_future_token_acc(
        cond_real_tokens: torch.Tensor,
        cond_prob_sample: torch.Tensor,
        path_mask: torch.Tensor
    ) -> np.ndarray:
    
    # remove from gpu
    cond_real_tokens.cpu()
    cond_prob_sample.cpu()
    path_mask.cpu()

    # class labels of the sampled model prediction
    cond_sample_tokens = torch.argmax(cond_prob_sample, dim=1)
    matches = []
    for ii, sample_pos in enumerate(path_mask):

        temp_real_tokens = cond_real_tokens[ii, sample_pos.nonzero()].squeeze(1)
        temp_sample_tokens = cond_sample_tokens[ii, sample_pos.nonzero()].squeeze(1)
        matches.append(
                (temp_real_tokens == temp_sample_tokens).tolist()
        )

    acc = []
    for match in matches:
        try: 
            acc.append(sum(match*1)/len(match))
        except ZeroDivisionError:
            acc.append(0)
        return np.mean(acc)

@torch.no_grad()
def compute_pos_entropy(probs: torch.Tensor) -> torch.Tensor:

    # average positional entropy
    pos_entropy = torch.mean(torch.mean(-probs * torch.log(probs), dim = 1), dim = 0)
    return pos_entropy


def elbo_objective(
        model: nn.Module,
        realization: torch.Tensor,
        args: any
        ) -> (
                torch.Tensor,
                float,
                float,
                float,
                torch.Tensor
        ):

            bs, channel, seq_length = realization.size()

            # get a batch of random sampling paths
            sampled_random_path = sample_random_path(bs, seq_length, device=args.device)
            # sample a set of random sampling steps for each individual training image in the current batch
            idx = sample_random_index_for_sampling(bs, seq_length, device=args.device, option='random')
            # we create a mask that masks the locations wher we've already sampled
            random_path_mask = create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
            # create a mask that masks the locations where are currently sampling
            current_path_mask = create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
            # future samplign locations (i.e. >t)
            future_path_mask = create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
            # tokenize realizations
            real_tokens, bs, seq_length = create_token_labels(args, realization)
            # mask realizations
            real_token_masked = mask_realizations(real_tokens, random_path_mask)
            # conditional probs
            conditional_prob, probs = predict_conditional_prob(model, real_token_masked, idx, args)
            # evaluate the value of the log prob for the given realization
            log_prob = log_prob_of_realization(args, conditional_prob, real_tokens)
            # compute an average over all the unsampled locations for each image in the batch
            #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob.to(args.device), real_token_masked.to(args.device))
            log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked)
            # compute an average over all the unsampled locations for each image in the batch
            log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, seq_length)
            # compute an average loss i.e. negative average log likelihood over teh batch elements
            loss = compute_average_loss_for_batch(log_prob_weighted)
            

            # compute metrics
            cond_prob_sample = sample_from_conditional(conditional_prob)
            acc = sample_recover(real_tokens, cond_prob_sample, current_path_mask)
            prev_acc = compute_prev_token_acc(real_tokens, cond_prob_sample, random_path_mask)
            future_acc = compute_future_token_acc(real_tokens, cond_prob_sample, future_path_mask)
            # average positional entropy
            pos_entropy = compute_pos_entropy(probs=probs)

            return (
                    loss,
                    acc,
                    prev_acc,
                    future_acc,
                    pos_entropy
            )


' model prediction with class conditional '
def cond_predict_conditional_prob(
        model: nn.Module,
        real_token_masked: torch.Tensor,
        y_c: torch.Tensor,
        idx: any,
        args: any
    ) -> (
            any,
            torch.Tensor
    ):
        #logits = model(x=real_token_masked.to(args.device), t=idx.view(-1,), y_c=y_c)
        logits = model(x=real_token_masked, t=idx.view(-1,), y_c=y_c)
        probs = F.softmax(
                logits,
                dim=1
        )

        conditional_prob = OneHotCategorical(probs=probs.permute(0,2,1))
  #     conditional_prob = Categorical(probs=probs.permute(0,2,1))
        
        return (
                conditional_prob,
                probs
        )


def cond_elbo_objective(
        model: nn.Module,
        realization: torch.Tensor,
        y_c: torch.Tensor,
        args: any,
        iteration: int
        ) -> (
                torch.Tensor,
                tuple
        ):

            bs, channel, seq_length = realization.size()

            # get a batch of random sampling paths
            sampled_random_path = sample_random_path(bs, seq_length, device=args.device)
            # sample a set of random sampling steps for each individual training samples in the current batch
            idx = sample_random_index_for_sampling(bs, seq_length, device=args.device, option='random')
            # we create a mask that masks the locations wher we've already sampled
            random_path_mask = create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
            # create a mask that masks the locations where are currently sampling
            current_path_mask = create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
            # future samplign locations (i.e. >t)
            future_path_mask = create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
            # tokenize realizations
            real_tokens, bs, seq_length = create_token_labels(args,realization)
            #real_tokens = realizations.clone().squeeze(1)
            # mask realizations
            real_token_masked = mask_realizations(real_tokens, random_path_mask)
            # conditional probs
            conditional_prob, probs = cond_predict_conditional_prob(model, real_token_masked, y_c, idx, args)
            # evaluate the value of the log prob for the given realization
            log_prob = log_prob_of_realization(args, conditional_prob, real_tokens)
            # compute an average over all the unsampled locations for each image in the batch
            #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob.to(args.device), real_token_masked.to(args.device))
            log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked)
            #log_prob_unsampled = log_prob_of_unsampled_locations(log_prob, real_token_masked, real_tokens)
            
            # compute an average over all the unsampled locations for each image in the batch
            log_prob_weighted = weight_log_prob(log_prob_unsampled, idx, seq_length)
            # compute an average loss i.e. negative average log likelihood over teh batch elements
            loss = compute_average_loss_for_batch(log_prob_weighted)
           
            # compute metrics
            if iteration % args.enter_eval == 0:
                
                
                with torch.no_grad():
                    
                    # compute accuracy given time position
                    sample_seq = torch.argmax(sample_from_conditional(conditional_prob), dim=1) # create numerical token sequences
                    
                    # convert to cpu
                    real_tokens = real_tokens.cpu()
                    sample_seq = sample_seq.cpu()
                    idx = idx.cpu()
                    sampled_random_path = sampled_random_path.cpu()
                    probs = probs.cpu()
                    
                    
                    prev_B_hard_acc, prev_B_soft_acc, fut_B_hard_acc, fut_B_soft_acc, current_B_hard_acc, current_B_soft_acc = eval_funcs.compute_acc_given_time_pos(
                        real_tokens=real_tokens,
                        sample_seq=sample_seq,
                        sample_path=sampled_random_path,
                        idx=idx
                    )

                    # copmute ppl given time position
                    current_ppl, prev_ppl, fut_ppl = eval_funcs.compute_ppl_given_time_pos(
                            probs=probs,
                            sample_path=sampled_random_path,
                            idx=idx
                    )

                    # average positional entropy
                    pos_entropy = compute_pos_entropy(probs=probs).mean().item()
                    

                    metric_evals = (
                            prev_B_hard_acc,
                            prev_B_soft_acc,
                            fut_B_hard_acc,
                            fut_B_soft_acc,
                            current_B_hard_acc,
                            current_B_soft_acc,
                            current_ppl,
                            prev_ppl,
                            fut_ppl,
                            pos_entropy
                    )

            else:
                metric_evals = (None)

            return (
                    loss,
                    metric_evals
            )



