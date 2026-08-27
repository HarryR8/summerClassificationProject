import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train_one_epoch(
    model: nn.Module, #the model
    loader: DataLoader, #data in batches
    criterion: nn.Module, #the loss function
    optimizer: torch.optim.Optimizer, #adjust the weights
    device: torch.device, #CPU/GPU
) -> dict:
    """
    Train for one epoch.

    Returns:
        {"loss": float, "auc": float}
    """
    model.train()

    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="Train", leave=False): #tqdm is a library for visual progress bar
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        #Pull the images and their true labels out of the batch, and send them to the GPU/CPU.

        optimizer.zero_grad() #Clear any leftover gradients from the previous batch
        logits = model(images) #raw scores straight out of the network (two numbers per image, one for noncancerous, one for cancerous)
        loss = criterion(logits, labels.long()) #loss function defined in scripts/train.py, logits==predictions, label==correct ans
        loss.backward() #Backpropagation, if the backbone is frozen, it only goes backwards through the head layers
        #if not frozen, goes through all layers
        optimizer.step() #updates the weights using what backpropagation just calculated

        total_loss += loss.item() * images.size(0) 

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy() #return a single probability
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    avg_loss = total_loss / len(all_labels)

    # Safety guard against single-class batches (can happen with tiny datasets)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {"loss": avg_loss, "auc": auc}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns:
        {
            "loss": float,
            "auc": float,
            "labels": np.ndarray,    # (N,) ground truth
            "probs": np.ndarray,     # (N,) P(cancerous)
            "image_ids": list[str],  # for error analysis
        }
    """
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_image_ids = []

    with torch.no_grad(): #same thing as in training but don't track gradients as it is eval
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels.long())

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_image_ids.extend(batch["image_id"])

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    avg_loss = total_loss / len(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss": avg_loss,
        "auc": auc,
        "labels": all_labels,
        "probs": all_probs,
        "image_ids": all_image_ids,
    }