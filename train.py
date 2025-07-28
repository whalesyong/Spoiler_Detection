import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os
import random
import numpy as np
from typing import Dict, Optional, Tuple

# set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertMLMTrainer: 
    """
    Trainer class for the BERT model supporting MLM pretraining ONLY, similar to how it was done in the original paper.

    The MLM task is as follows:
    During training, we see each text sample, and mask select 15% of all tokens.
    Of these 15% tokens of the training data, we replace them with:
    1. [MASK] token, 80% of the time. 
    2. A random token 10% of the time.
    3. the unchanged token 10% of the time. 
    Then, we will predict this token with CrossEntropyLoss
 
    """
    def __init__(
        self, 
        model, 
        tokenizer,
        train_dataloader, 
        val_dataloader=None,
        lr=1e-4,
        weight_decay=1e-4,
        warmup_steps=10000,
        max_grad_norm=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='./checkpoints', 
        mask_prob=0.15
    ): 
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.mask_prob = mask_prob
        self.warmup_steps = warmup_steps

        os.makedirs(save_dir, exist_ok=True)

        # bert paper uses AdamW
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
        )
        
        self.scheduler = self._get_linear_schedule_with_warmup()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100) # default ignore_index for HF datasets
        
    def _get_linear_schedule_with_warmup(self):
        # create lr scheduler with linear warmup, decay. 
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(len(self.train_dataloader) - current_step) / 
                float(max(1, len(self.train_dataloader) - self.warmup_steps))
            )
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # token masker. 
    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prepare masked tokens inputs, labels for MLM. This function will do the following: 
        1. Clone the input (should be a 2D Tensor, with (batch_size, seq_len))
        2. For each data point in the batch, perform masking using the criteria specified above. Special tokens
        should not be masked out, since they are important to us during training. 
            
        """

        if self.tokenizer.mask_token is None: 
            raise ValueError("tokenizer has NO mask token.")
        labels = inputs.clone() 
        probability_matrix = torch.full(inputs.shape, self.mask_prob)

        # zero out the probabilities of special tokens. 
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # set prob to 0 for all special tokens 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # for each token, we draw from a bernoulli (binary) distribution to decide if the token is selected for masking. 
        # we obtain this value from the probability matrix. 
        # here, should be shape (batch_size, seq_len)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # all non masked indices will be set to -100, since we only want to compare the masked tokens from the labels matrix. 
        # -100 is the value we gave to self.criterion. 
        labels[~masked_indices] = -100

        # for all the masked indices, we replace them with the [MASK] token 80% of the time 
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id


        # from the remaining that are not replaced (20% chance), replace them with a 50% chance with a random token. 
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced 
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=self.device)
        inputs[indices_random] = random_words[indices_random]

        # the other 10%, we just do nothing to the tokens 
        return inputs, labels



    def train_epoch(self):

        # train for one epoch
        self.model.train() 
        total_loss = 0
        total_correct = 0
        total_predictions = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training")

        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device) 
            attention_mask = batch['attention_mask'].to(self.device) 

                # apply mask 
            masked_input_ids, labels = self.mask_tokens(input_ids)

            self.optimizer.zero_grad()

            # forward 
            outputs = self.model(
                input_ids=masked_input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )

            loss = outputs['loss']
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

                # compute acc for masked tokens 
            predictions = outputs['logits'].argmax(dim=-1)
            mask = labels != -100
            correct = (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()
            total_correct += correct 


            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/mask.sum().item():.4f}' if mask.sum().item() > 0 else '0.0000',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / len(self.train_dataloader)
        avg_accuracy = total_correct / total_predictions if total_predictions > 0 else 0

        return {
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy
        }
    def validate(self): 
            # validation. counts CE loss, and number of correct instances 
        if self.val_dataloader is None: 
            print("Warning: No data loader given. No validation")
            return {}
        self.model.eval()
        total_loss, total_correct, total_predictions = 0,0,0
        with torch.no_grad(): 
            progress_nar = tqdm(self.val_dataloader, desc="Validation")

            for batch in progress_bar: 
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # apply mask 
                masked_input_ids, labels = self.mask_tokens(input_ids)

                outputs = self.model(
                    input_ids = masked_input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs['loss']
                total_loss += loss.item()
                    
                
                predictions = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                correct = (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()
                total_correct += correct
                    
                progress_bar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'val_acc': f'{correct/mask.sum().item():.4f}' if mask.sum().item() > 0 else '0.0000'
                })
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
        
    
    def train(self, num_epochs, save_every=1, eval_every=1):

        # this is the main training loop
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
            
            # Validate
            val_metrics = {}
            if epoch % eval_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                               f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Save checkpoint
            if epoch % save_every == 0:
                is_best = val_metrics.get('val_loss', float('inf')) < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        logger.info("Training completed!")        






            
            
            
            
            
            
            
    
        
    
        