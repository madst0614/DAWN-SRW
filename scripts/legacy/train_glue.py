"""
DAWN GLUE Benchmark Fine-tuning Script

Usage:
    python train_glue.py --task sst2 --checkpoint path/to/model.pt
    python train_glue.py --task all --checkpoint path/to/model.pt
    python train_glue.py --task sst2 --checkpoint path/to/model.pt --collect_neurons

GLUE Tasks:
    - SST-2: Sentiment Analysis (2 classes)
    - MNLI: Natural Language Inference (3 classes)
    - QQP: Quora Question Pairs (2 classes)
    - QNLI: Question NLI (2 classes)
    - RTE: Recognizing Textual Entailment (2 classes)
    - MRPC: Microsoft Research Paraphrase Corpus (2 classes)
    - CoLA: Corpus of Linguistic Acceptability (2 classes, Matthews corr)
    - STS-B: Semantic Textual Similarity (regression)
"""

import os
import sys
import json
import argparse
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model_by_version
from datasets import load_dataset
from transformers import AutoTokenizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

# ============================================================
# GLUE Task Configurations
# ============================================================

GLUE_TASKS = {
    'sst2': {
        'dataset': 'glue',
        'subset': 'sst2',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_fields': ['sentence'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 3,
        'batch_size': 128,
    },
    'mnli': {
        'dataset': 'glue',
        'subset': 'mnli',
        'num_labels': 3,
        'metric': 'accuracy',
        'text_fields': ['premise', 'hypothesis'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 3,
        'batch_size': 128,
    },
    'qqp': {
        'dataset': 'glue',
        'subset': 'qqp',
        'num_labels': 2,
        'metric': 'f1',
        'text_fields': ['question1', 'question2'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 3,
        'batch_size': 128,
    },
    'qnli': {
        'dataset': 'glue',
        'subset': 'qnli',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_fields': ['question', 'sentence'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 3,
        'batch_size': 128,
    },
    'rte': {
        'dataset': 'glue',
        'subset': 'rte',
        'num_labels': 2,
        'metric': 'accuracy',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 5,
        'batch_size': 64,
    },
    'mrpc': {
        'dataset': 'glue',
        'subset': 'mrpc',
        'num_labels': 2,
        'metric': 'f1',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 5,
        'batch_size': 64,
    },
    'cola': {
        'dataset': 'glue',
        'subset': 'cola',
        'num_labels': 2,
        'metric': 'matthews',
        'text_fields': ['sentence'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 5,
        'batch_size': 64,
    },
    'stsb': {
        'dataset': 'glue',
        'subset': 'stsb',
        'num_labels': 1,  # regression
        'metric': 'pearson',
        'text_fields': ['sentence1', 'sentence2'],
        'label_field': 'label',
        'lr': 3e-4,
        'epochs': 5,
        'batch_size': 64,
    },
}


# ============================================================
# DAWN Classifier Wrapper
# ============================================================

class DAWNForSequenceClassification(nn.Module):
    """DAWN model with classification head for GLUE tasks"""

    def __init__(self, dawn_model, num_labels, dropout=0.1):
        super().__init__()
        self.dawn = dawn_model
        self.num_labels = num_labels
        self.d_model = dawn_model.d_model

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.d_model, num_labels)

        # Neuron activation collection
        self.collect_neurons = False
        self.neuron_activations = []

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, labels=None, return_routing_info=False):
        """
        Args:
            input_ids: [B, S]
            labels: [B] for classification, [B] for regression
        Returns:
            loss, logits, (routing_infos)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Single forward pass through DAWN layers (without lm_head)
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = self.dawn.token_emb(input_ids) + self.dawn.pos_emb(positions)

        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        mask = ~mask.unsqueeze(0).unsqueeze(0)

        routing_infos = []
        for layer in self.dawn.layers:
            x, routing_info = layer(x, mask)
            routing_infos.append(routing_info)

        x = self.dawn.norm(x)

        # [CLS] token representation (first token)
        cls_output = x[:, 0, :]  # [B, d_model]

        # Classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [B, num_labels]

        # Collect neuron activations if enabled
        if self.collect_neurons:
            self._collect_neuron_activations(routing_infos)

        # Compute loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression (STS-B)
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)

        if return_routing_info:
            return loss, logits, routing_infos
        return loss, logits

    def _collect_neuron_activations(self, routing_infos):
        """Collect neuron weights from routing info"""
        batch_activations = []
        for layer_info in routing_infos:
            attn_weights = layer_info['attention']['neuron_weights']  # [B, n_compress]
            mem_weights = layer_info['memory']['neuron_weights']  # [B, n_compress]
            batch_activations.append({
                'attention': attn_weights.detach().cpu(),
                'memory': mem_weights.detach().cpu(),
            })
        self.neuron_activations.append(batch_activations)

    def get_neuron_activations(self):
        """Return collected neuron activations and clear buffer"""
        activations = self.neuron_activations
        self.neuron_activations = []
        return activations

    def enable_neuron_collection(self, enable=True):
        """Enable/disable neuron activation collection"""
        self.collect_neurons = enable


# ============================================================
# Dataset
# ============================================================

class GLUEDataset(Dataset):
    """Dataset wrapper for GLUE tasks"""

    def __init__(self, data, tokenizer, task_config, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.task_config = task_config
        self.max_length = max_length
        self.text_fields = task_config['text_fields']
        self.label_field = task_config['label_field']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get text(s)
        if len(self.text_fields) == 1:
            text = item[self.text_fields[0]]
        else:
            text = ' [SEP] '.join([item[f] for f in self.text_fields])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        label = item[self.label_field]

        return {
            'input_ids': input_ids,
            'label': torch.tensor(label) if not isinstance(label, float) else torch.tensor(label, dtype=torch.float),
        }


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_metrics(preds, labels, task_name):
    """Compute task-specific metrics"""
    task_config = GLUE_TASKS[task_name]
    metric = task_config['metric']

    if metric == 'accuracy':
        return {'accuracy': accuracy_score(labels, preds)}
    elif metric == 'f1':
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds),
        }
    elif metric == 'matthews':
        return {
            'accuracy': accuracy_score(labels, preds),
            'matthews_corr': matthews_corrcoef(labels, preds),
        }
    elif metric == 'pearson':
        return {
            'pearson': pearsonr(preds, labels)[0],
            'spearman': spearmanr(preds, labels)[0],
        }
    else:
        return {'accuracy': accuracy_score(labels, preds)}


# ============================================================
# Training Loop
# ============================================================

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            loss, logits = model(input_ids, labels=labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def evaluate(model, eval_loader, task_name, device, collect_neurons=False):
    """Evaluate model on validation set"""
    model.eval()
    task_config = GLUE_TASKS[task_name]

    if collect_neurons:
        model.enable_neuron_collection(True)

    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            loss, logits = model(input_ids, labels=labels)

            if task_config['num_labels'] == 1:
                # Regression
                preds = logits.squeeze().cpu().numpy()
            else:
                # Classification
                preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

    if collect_neurons:
        model.enable_neuron_collection(False)
        neuron_activations = model.get_neuron_activations()
    else:
        neuron_activations = None

    metrics = compute_metrics(all_preds, all_labels, task_name)
    metrics['loss'] = total_loss / num_batches

    return metrics, neuron_activations


# ============================================================
# Main Training Function
# ============================================================

def train_glue_task(
    task_name,
    checkpoint_path,
    output_dir,
    model_version='12.0',
    collect_neurons=False,
    device='cuda',
    use_amp=True,
):
    """Train DAWN on a GLUE task"""
    print(f"\n{'='*60}")
    print(f"Training DAWN on GLUE task: {task_name.upper()}")
    print(f"{'='*60}")

    task_config = GLUE_TASKS[task_name]

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load dataset
    print(f"Loading {task_name} dataset...")
    dataset = load_dataset(task_config['dataset'], task_config['subset'])

    train_data = dataset['train']
    if task_name == 'mnli':
        eval_data = dataset['validation_matched']
    else:
        eval_data = dataset['validation']

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # Create datasets
    max_length = 128
    train_dataset = GLUEDataset(train_data, tokenizer, task_config, max_length)
    eval_dataset = GLUEDataset(eval_data, tokenizer, task_config, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=task_config['batch_size'],
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=task_config['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Load DAWN checkpoint
    print(f"\nLoading DAWN checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
    if 'model_config' in model_config:
        model_config = model_config['model_config']

    # Update vocab_size for BERT tokenizer
    model_config['vocab_size'] = tokenizer.vocab_size

    print(f"Model config: {model_config}")

    # Create DAWN model
    dawn_model = create_model_by_version(model_version, model_config)

    # Load pretrained weights (skip embedding if vocab size differs)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Filter out mismatched keys
    model_state = dawn_model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                print(f"  Skipping {k}: shape mismatch ({v.shape} vs {model_state[k].shape})")
        else:
            print(f"  Skipping {k}: not in model")

    dawn_model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)}/{len(state_dict)} parameters")

    # Create classifier model
    model = DAWNForSequenceClassification(
        dawn_model,
        num_labels=task_config['num_labels'],
        dropout=0.1,
    )
    model = model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=task_config['lr'],
        weight_decay=0.01,
    )

    # Scheduler
    num_training_steps = len(train_loader) * task_config['epochs']
    num_warmup_steps = int(num_training_steps * 0.1)

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    scaler = GradScaler() if use_amp else None

    # Training loop
    best_metric = -float('inf')
    best_epoch = 0
    results = []

    print(f"\nTraining for {task_config['epochs']} epochs...")
    print(f"Batch size: {task_config['batch_size']}")
    print(f"Learning rate: {task_config['lr']}")

    for epoch in range(task_config['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{task_config['epochs']} ---")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        collect = collect_neurons and (epoch == task_config['epochs'] - 1)
        metrics, neuron_acts = evaluate(model, eval_loader, task_name, device, collect_neurons=collect)

        print(f"Eval loss: {metrics['loss']:.4f}")
        for k, v in metrics.items():
            if k != 'loss':
                print(f"  {k}: {v:.4f}")

        # Track best
        main_metric = task_config['metric']
        metric_value = metrics.get(main_metric, metrics.get('accuracy', 0))

        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch + 1

            # Save best model
            save_path = os.path.join(output_dir, f'{task_name}_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'task_name': task_name,
                'epoch': epoch + 1,
                'metrics': metrics,
                'model_config': model_config,
            }, save_path)
            print(f"  Saved best model: {save_path}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
        })

    # Save final results
    results_path = os.path.join(output_dir, f'{task_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'task': task_name,
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'results': results,
            'config': task_config,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task {task_name.upper()} completed!")
    print(f"Best {main_metric}: {best_metric:.4f} (epoch {best_epoch})")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")

    # Save neuron activations if collected
    if neuron_acts is not None:
        neuron_path = os.path.join(output_dir, f'{task_name}_neurons.pt')
        torch.save(neuron_acts, neuron_path)
        print(f"Neuron activations saved to: {neuron_path}")

    return {
        'task': task_name,
        'best_metric': best_metric,
        'best_epoch': best_epoch,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN GLUE Fine-tuning')
    parser.add_argument('--task', type=str, required=True,
                        help='GLUE task name or "all" for all tasks')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint directory or best_model.pt file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {checkpoint_dir}/glue)')
    parser.add_argument('--model_version', type=str, default='17.1',
                        help='DAWN model version')
    parser.add_argument('--collect_neurons', action='store_true',
                        help='Collect neuron activations during final eval')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')

    args = parser.parse_args()

    # Handle checkpoint path: directory or file
    checkpoint_path = args.checkpoint
    if os.path.isdir(checkpoint_path):
        # Directory provided - look for best_model.pt
        checkpoint_dir = checkpoint_path
        checkpoint_file = os.path.join(checkpoint_path, 'best_model.pt')
        if not os.path.exists(checkpoint_file):
            # Try other common names
            for name in ['best_model.pt', 'model.pt', 'checkpoint.pt']:
                candidate = os.path.join(checkpoint_path, name)
                if os.path.exists(candidate):
                    checkpoint_file = candidate
                    break
            else:
                raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
        checkpoint_path = checkpoint_file
    else:
        # File provided
        checkpoint_dir = os.path.dirname(checkpoint_path)

    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(checkpoint_dir, 'glue')

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run tasks
    if args.task.lower() == 'all':
        tasks = list(GLUE_TASKS.keys())
    else:
        tasks = [args.task.lower()]

    all_results = []
    for task in tasks:
        if task not in GLUE_TASKS:
            print(f"Unknown task: {task}, skipping...")
            continue

        result = train_glue_task(
            task_name=task,
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            model_version=args.model_version,
            collect_neurons=args.collect_neurons,
            device=args.device,
            use_amp=not args.no_amp,
        )
        all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("GLUE BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")
    for result in all_results:
        print(f"{result['task'].upper():8s}: {result['best_metric']:.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
