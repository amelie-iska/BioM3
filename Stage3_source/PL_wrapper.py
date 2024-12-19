import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import OneHotCategorical
from transformers.optimization import Adafactor

# PL functions
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import functools
import math
#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

from sklearn.model_selection import train_test_split

from Stage3_source.DSEma import moving_average, clone_zero_model
import Stage3_source.transformer_training_helper as trainer_tools
import Stage3_source.helper_funcs as helper_tools
import Stage3_source.eval_metrics as eval_funcs
import Stage3_source.preprocess as prep

import copy

from torch.utils.data import DataLoader
import pandas as pd

from transformers import get_cosine_schedule_with_warmup

class PL_ProtARDM(pl.LightningModule):


    def __init__(
            self, 
            args: any,
            model: nn.Module,
            #ema_model: nn.Module,
        ):

        super().__init__()
        #self.save_hyperparameters()

        # arguments
        self.script_args = args
        
        # the whole model
        self.model = model
        #self.ema_model = ema_model

        #clone_zero_model(self.model, self.ema_model, zero_stage=3)
        ##self.ema_model = copy.deepcopy(self.model)

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y_c: torch.Tensor,
            ema=False,
        ) -> torch.Tensor:
    
        if ema:
            logits = self.ema_model(x=x, t=t.view(-1,), y_c=y_c)
        else:
            logits = self.model(x=x, t=t.view(-1,), y_c=y_c)
        return logits
        #return F.softmax(logits, dim=1)
    

    #def on_train_batch_end(self, *args, **kwargs):
    #    clone_zero_model(self.model, self.ema_model, zero_stage=3)
    #    #moving_average(self.model, self.ema_model, beta=0.0, zero_stage=3)


    def configure_optimizers(self, ):
        
        if self.script_args.choose_optim == 'AdamW':

            if isinstance(self, FSDP):
                print("Enter FSDP")
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.script_args.lr, weight_decay=self.script_args.weight_decay)

            else:
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.script_args.lr, weight_decay=self.script_args.weight_decay)

        elif self.script_args.choose_optim == 'AdaFactor':
            optimizer = Adafactor(self.parameters(), lr=self.script_args.lr, weight_decay=self.script_args.weight_decay, relative_step=False)

        elif self.script_args.choose_optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.script_args.lr)
        
        elif self.script_args.choose_optim == 'DeepSpeedCPUAdam':
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.script_args.lr, weight_decay=self.script_args.weight_decay)
        
        if self.script_args.scheduler_gamma is not None:
            if isinstance(self.script_args.scheduler_gamma, str):
                if 'coswarmup' == self.script_args.scheduler_gamma.lower():
                    print(f'Using cossine warmup scheduler with decay')
                    num_warmup_steps=self.script_args.traindata_len
                    num_training_steps=self.script_args.traindata_len*self.script_args.epochs
                    print(f'Num_warmup_steps={num_warmup_steps}')
                    print(f'Num_training_steps={num_training_steps}')

                    def _get_cosine_schedule_with_warmup_lr_lambda(
                        current_step: int, num_warmup_steps: int, num_training_steps: int, num_cycles: float
                    ):
                        if current_step < num_warmup_steps:
                            return float(current_step) / float(max(1, num_warmup_steps))
                        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
                    
                    lr_lambda = functools.partial(
                        _get_cosine_schedule_with_warmup_lr_lambda,
                        num_warmup_steps=num_warmup_steps,
                        num_training_steps=num_training_steps,
                        num_cycles=0.5,
                    )
                    return {
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1),
                            "interval": "step",
                        },
                    }

                    #return {
                    #    "optimizer": optimizer,
                    #    "lr_scheduler": {
                    #        "scheduler": get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps),
                    #        "interval": "step",
                    #    },
                    #}
            else:
                print(f'Using Exponential learning rate decay / epoch with factor: {self.script_args.scheduler_gamma}')
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.script_args.scheduler_gamma, verbose=True),
                        "interval": "epoch",
                    },
                }
        else:
            return optimizer

        #else:
        #    print("Please make choose_option variable from these options: 'AdamW', 'AdaFactor', 'Adam', 'DeepSpeedCPUAdam'")

    def common_step(
            self,
            realization: torch.Tensor,
            realization_idx: any,
            stage: str) -> dict:

        if isinstance(realization, list):

            # class labels
            y_c = realization[1]#.long()

            # input samples 
            realization = realization[0]
            batch_size, seq_length = realization.size()

        realization = realization.reshape(batch_size, 1, seq_length).long()
        
        train_tuple = self.cond_elbo_objective(
                realization=realization,
                y_c=y_c,
                realization_idx=realization_idx,
                stage=stage,
                ema=True if 'ema' in stage.lower() else False,
        )
        
        if len(train_tuple) == 1:
            loss = train_tuple[0]
        else:
            loss = train_tuple[0]
            metrics = train_tuple[1]
               
        if realization_idx == 0:
            gpu_memory_usage = helper_tools.print_gpu_initialization()
            self.log(f"{stage}_gpu_memory_usage", gpu_memory_usage, sync_dist=True)
    
        sync_dist = True if 'val' in stage else False
        # track loss 
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=sync_dist)
        # track performance metrics
        if len(train_tuple) > 1:
            self.log(f"{stage}_prev_hard_acc", metrics[0], prog_bar=True,  on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_prev_soft_acc", metrics[1], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_fut_hard_acc", metrics[2], prog_bar=True, on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_fut_soft_acc", metrics[3], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_current_hard_acc", metrics[4], prog_bar=True, on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_current_soft_acc", metrics[5], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_current_ppl", metrics[6], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_prev_ppl", metrics[7], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_fut_ppl", metrics[8], on_step=True, on_epoch=True, sync_dist=sync_dist)
            self.log(f"{stage}_pos_entropy", metrics[9], on_step=True, on_epoch=True, sync_dist=sync_dist)

        torch.cuda.empty_cache()
        return {'loss': loss}

    def training_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        return self.common_step(realization, realization_idx, stage='train')

    def validation_step(
            self,
            realization: torch.Tensor,
            realization_idx: any):
        self.common_step(realization, realization_idx, stage='val')
        #self.common_step(realization, realization_idx, stage='EMA_val')

    def apply_OneHotCat(self, probs: torch.Tensor) -> any:
        return OneHotCategorical(probs=probs.permute(0,2,1))
        #return OneHotCategorical(probs=F.softmax(probs.permute(0,2,1), dim=-1))

    def cond_elbo_objective(
            self,
            realization: torch.Tensor,
            y_c: torch.Tensor,
            realization_idx: any,
            stage: str,
            ema=False,
        ):

            bs, channel, seq_length = realization.size()

            # get a batch of random sampling paths
            sampled_random_path = trainer_tools.sample_random_path(bs, seq_length, device=self.script_args.device)
            # sample a set of random smapling steps for each individual training sequences in the current batch
            idx = trainer_tools.sample_random_index_for_sampling(bs, seq_length, device=self.script_args.device, option='random')
            # we create a mask that masks the location were we've already sampled
            random_path_mask = trainer_tools.create_mask_at_random_path_index(sampled_random_path, idx, bs, seq_length)
            # create a mask that masks the location where we are currently sampling
            current_path_mask = trainer_tools.create_sampling_location_mask(sampled_random_path, idx, bs, seq_length)
            # future sampling locations (i.e. >t)
            future_path_mask = trainer_tools.create_mask_at_future_path_index(sampled_random_path, idx, bs, seq_length)
            # tokenize realization
            real_tokens, bs, seq_length = trainer_tools.create_token_labels(self.script_args, realization)
            #real_tokens = realization.clone().squeeze(1)
            # mask realizations
            real_token_masked = trainer_tools.mask_realizations(real_tokens, random_path_mask)
            # conditional probs
            #probs = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)   
            logits = self(x=real_token_masked, t=idx, y_c=y_c, ema=ema)   

            conditional_prob = OneHotCategorical(logits=logits.permute(0,2,1))
            #conditional_prob = self.apply_OneHotCat(probs=probs)
            # evaluate the value of the log prob for the given realization
            log_prob = trainer_tools.log_prob_of_realization(self.script_args, conditional_prob, real_tokens)

            # compute an average over all the unsampled
            #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob.to(self.script_args.device), real_token_masked.to(self.script_args.device))
            log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked)
            #log_prob_unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, real_token_masked, real_tokens)


            # compute an average loss i.e. negative average log-likelihood over the batch elements
            log_prob_weighted = trainer_tools.weight_log_prob(log_prob_unsampled, idx, seq_length)
            # compute an average loss i.e. negative average log-likelihood over the batch elements
            loss = trainer_tools.compute_average_loss_for_batch(log_prob_weighted)
            
            #if 'val' in stage:
            probs = F.softmax(logits, dim=1)
            metrics = self.performance_step(
                        real_tokens=real_tokens.cpu(),
                        idx=idx.cpu(),
                        sampled_random_path=sampled_random_path.cpu().float(),
                        probs=probs.cpu().float(),
                        conditional_prob=conditional_prob)
                    
            return loss, metrics


           # return loss,
    
    @torch.no_grad()
    def performance_step(
                    self,
                    real_tokens: torch.Tensor,
                    idx: torch.Tensor,
                    sampled_random_path: torch.Tensor,
                    probs: torch.Tensor,
                    conditional_prob: torch.Tensor
                    ) -> tuple:


        # create numerical token sequence
        sample_seq = torch.argmax(trainer_tools.sample_from_conditional(conditional_prob).cpu(), dim=1)
       
        # eval prev positions in terms of time 
        prev_B_hard_acc, prev_B_soft_acc, fut_B_hard_acc, fut_B_soft_acc, current_B_hard_acc, current_B_soft_acc = eval_funcs.compute_acc_given_time_pos(
                real_tokens=real_tokens,
                sample_seq=sample_seq,
                sample_path=sampled_random_path,
                idx=idx
        )
        
        # compute ppl given time position
        current_ppl, prev_ppl, fut_ppl = eval_funcs.compute_ppl_given_time_pos(
                probs=probs,
                sample_path=sampled_random_path,
                idx=idx
        )

        # average positional entropy
        pos_entropy = trainer_tools.compute_pos_entropy(probs=probs).mean().item()
            
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
        
        return metric_evals



class PFamDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        #df = pd.read_csv(args.data_root)
        #data = torch.load(args.data_root)
        data = self.load_data()

        num_seq_list, text_emb_list = prep.prepare_protein_data(
                args=args,
                data_dict=data
        )
        
        print('Performing 80/20 random train/val split')
        num_seq_list_train, num_seq_list_val, text_emb_train, text_emb_val = train_test_split(num_seq_list,
                                                                                            text_emb_list,
                                                                                            test_size=args.valid_size,
                                                                                            #stratify=class_label_list,
                                                                                            random_state=args.seed)
        print(f'Number of training samples: {len(num_seq_list_train)}')
        print(f'Number of validation samples: {len(num_seq_list_val)}')
              
        self.train_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_train,
            text_emb=text_emb_train
        )
        
        self.val_dataset = prep.protein_dataset(
            num_seq_list=num_seq_list_val,
            text_emb=text_emb_val
        )
    
    def load_data(self):
        
        try:
            
            print(self.args.swissprot_data_root, self.args.pfam_data_root)

            if self.args.swissprot_data_root != "None":
                swissprot_data = torch.load(self.args.swissprot_data_root)
            else:
                swissprot_data=None
            
            if self.args.pfam_data_root != "None":
                pfam_data = torch.load(self.args.pfam_data_root)
            else:
                pfam_data=None

            if (self.args.swissprot_data_root != "None") and (self.args.pfam_data_root != "None"):
                return self.merge_and_append_values(dict1=swissprot_data, dict2=pfam_data)
            elif self.args.swissprot_data_root == "None":
                return pfam_data
            elif self.args.pfam_data_root == "None":
                return swissprot_data
            else:
                raise ValueError('Both SwissProt and Pfam datasets are unavailable.')
   
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {e}")


    def merge_and_append_values(self, dict1, dict2):
        
        merged_dict = {}

        # Combine all keys from both dictionaries
        all_keys = set(dict1) | set(dict2)

        for key in all_keys:
            values = []
            if key in dict1:
                values.append(dict1[key])
            if key in dict2:
                values.append(dict2[key])
            
            # Merge values for each key
            # This merges lists or appends non-list values
            merged_dict[key] = [item for sublist in values for item in (sublist if isinstance(sublist, list) else [sublist])]
        
        return merged_dict
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True
    )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
    )
