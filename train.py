import torch 
import torch.nn 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
import logging import os 
from typing import Dict, Optional, Typle

# set up logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertMLMTrainer: 
    """
    Trainer class for the BERT model supporting MLM pretraining. 
    """
    def __init__(
        self, 
        model, 
        train_dataloader, 
        val_dataloader=None,
        lr=1e-4,
        weight_decay=1e-4,
        warmup_steps=10000,
        max_grad_norm=1.0,
        device='cuda' if torch.cuda().is_available() else 'cpu',
        save_dir='./checkpoints'
    ): 
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # bert paper uses AdamW
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
        )

    
        