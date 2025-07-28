import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import BertForMaskedLM
from train import BertMLMTrainer
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader 
import resource
import time
import json
import csv
from datetime import datetime

# Set environment variables to reduce resource usage
os.environ['HF_DATASETS_NUM_PROC'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Try to increase file descriptor limit (may fail on HPC)
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
except:
    pass 

# Configuration
TOKENIZER_PATH = "bpe_tokenizer_with_special"
NUM_EPOCHS = 30
DATA_NAME = "BookCorpus_tokenized_hf_merged"
EMBED_DIM = 768
VOCAB_SIZE = 30000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 5000
SAVE_DIR = "./model_checkpoints"
BATCH_SIZE = 128

# Logging configuration
LOG_DIR = "./training_logs"

def setup_logging(local_rank):
    if local_rank == 0:  # Only main process creates directories
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log files
        train_log_file = os.path.join(LOG_DIR, f"training_metrics_{timestamp}.csv")
        val_log_file = os.path.join(LOG_DIR, f"validation_metrics_{timestamp}.csv")
        config_log_file = os.path.join(LOG_DIR, f"config_{timestamp}.json")
        
        # Write CSV headers
        with open(train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'learning_rate', 'timestamp'])
            
        with open(val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'val_loss', 'val_accuracy', 'timestamp'])
        
        # Save configuration
        config = {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'warmup_steps': WARMUP_STEPS,
            'embed_dim': EMBED_DIM,
            'vocab_size': VOCAB_SIZE,
            'tokenizer_path': TOKENIZER_PATH,
            'data_name': DATA_NAME
        }
        
        with open(config_log_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        return train_log_file, val_log_file, timestamp
    else:
        return None, None, None

def log_training_metrics(log_file, epoch, train_loss, train_accuracy, lr):
    if log_file is not None:  # Only main process logs
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_accuracy, lr, timestamp])

def log_validation_metrics(log_file, epoch, val_loss, val_accuracy):
    if log_file is not None:  # Only main process logs
        timestamp = datetime.now().isoformat()
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, val_loss, val_accuracy, timestamp])

class LoggingBertMLMTrainer(BertMLMTrainer):
    
    def __init__(self, train_log_file=None, val_log_file=None, local_rank=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_log_file = train_log_file
        self.val_log_file = val_log_file
        self.local_rank = local_rank
        self.is_main_process = local_rank == 0
        
    def train(self, num_epochs, save_every=1, eval_every=1):
        if self.is_main_process:
            print(f"Starting training for {num_epochs} epochs with logging...")
        
        # Original logging setup from parent class
        if self.is_main_process:
            print(f"Device: {self.device}")
            print(f"Training samples: {len(self.train_dataloader.dataset)}")
            if self.val_dataloader:
                print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            if self.is_main_process:
                print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Set epoch for distributed sampler
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch - 1)  # 0-indexed
            
            # Train epoch (uses parent class method)
            train_metrics = self.train_epoch()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Synchronize training metrics across processes for DDP
            if dist.is_initialized():
                train_loss_tensor = torch.tensor(train_metrics['train_loss'], device=self.device)
                train_acc_tensor = torch.tensor(train_metrics['train_accuracy'], device=self.device)
                
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
                
                train_metrics['train_loss'] = (train_loss_tensor / dist.get_world_size()).item()
                train_metrics['train_accuracy'] = (train_acc_tensor / dist.get_world_size()).item()
            
            if self.is_main_process:
                print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
                
                # Log training metrics
                log_training_metrics(
                    self.train_log_file, 
                    epoch, 
                    train_metrics['train_loss'], 
                    train_metrics['train_accuracy'], 
                    current_lr
                )
            
            # Validate
            val_metrics = {}
            if epoch % eval_every == 0:
                val_metrics = self.validate()
                
                # Synchronize validation metrics across processes for DDP
                if val_metrics and dist.is_initialized():
                    val_loss_tensor = torch.tensor(val_metrics['val_loss'], device=self.device)
                    val_acc_tensor = torch.tensor(val_metrics['val_accuracy'], device=self.device)
                    
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
                    
                    val_metrics['val_loss'] = (val_loss_tensor / dist.get_world_size()).item()
                    val_metrics['val_accuracy'] = (val_acc_tensor / dist.get_world_size()).item()
                
                if val_metrics and self.is_main_process:
                    print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
                    
                    # Log validation metrics
                    log_validation_metrics(
                        self.val_log_file,
                        epoch,
                        val_metrics['val_loss'],
                        val_metrics['val_accuracy']
                    )
            
            # Save checkpoint (only main process, using parent class method)
            if epoch % save_every == 0 and self.is_main_process:
                all_metrics = {**train_metrics, **val_metrics}
                is_best = val_metrics.get('val_loss', float('inf')) < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        if self.is_main_process:
            print("Training completed!")

def setup_ddp():
    """Initialize DDP environment"""
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device for this process
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return local_rank, device

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    # Setup DDP
    local_rank, device = setup_ddp()
    
    # Only print from rank 0 to avoid duplicate output
    is_main_process = local_rank == 0
    
    # Setup logging
    train_log_file, val_log_file, timestamp = setup_logging(local_rank)
    
    if is_main_process:
        print("Starting DDP training with logging...")
        if timestamp:
            print(f"Log files will be saved with timestamp: {timestamp}")
    
    # Load tokenizer on all processes (lightweight)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    # Load dataset sequentially to avoid resource conflicts
    if is_main_process:
        print("Loading dataset...")
        dataset = load_from_disk(DATA_NAME)
        dataset.set_format(type='torch', columns=["input_ids", "attention_mask"])
        print(f"Dataset loaded with {len(dataset)} samples")
    else:
        dataset = None
    # Wait for rank 0 to finish
    dist.barrier()
    
    # Broadcast the dataset object
    obj_list = [dataset]
    dist.broadcast_object_list(obj_list, src=0)
    dataset = obj_list[0]
    
    # Split data into train and val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    # Set dataset format for PyTorch
    train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask"])
    val_dataset.set_format(type='torch', columns=["input_ids", "attention_mask"])
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=dist.get_rank(),
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=dist.get_world_size(), 
        rank=dist.get_rank(),
        shuffle=False
    )
    
    # Create dataloaders with distributed samplers
    # Note: We don't need DataCollatorForLanguageModeling since the original trainer handles masking
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model and move to device
    model = BertForMaskedLM(EMBED_DIM, VOCAB_SIZE)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # Create trainer with logging (using the existing trainer interface)
    trainer = LoggingBertMLMTrainer(
        train_log_file=train_log_file,
        val_log_file=val_log_file,
        local_rank=local_rank,
        model=model, 
        tokenizer=tokenizer,
        train_dataloader=train_loader, 
        val_dataloader=val_loader, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY, 
        warmup_steps=WARMUP_STEPS,
        device=device,
        save_dir=SAVE_DIR
    )
    
    # Train the model
    trainer.train(NUM_EPOCHS)
    
    # Clean up
    cleanup_ddp()

if __name__ == "__main__":
    main()