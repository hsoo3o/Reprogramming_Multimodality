

import torch
from tqdm import tqdm
import open_clip
from preprocessing.preprocessing_utils import get_text_preprocess

def zeroshot_classifier(model, classnames, templates, bpe_path, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = get_text_preprocess(texts, bpe_path).to(device) #tokenize
            class_embeddings = model.forward({'text': texts}) #embed with text encoder
            class_embedding = class_embeddings['text']
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            class_embedding = class_embedding.mean(dim=0)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    return zeroshot_weights




def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def top1_top5_acc(logits, target):
    top1, top5, n = 0., 0., 0.

    # measure accuracy  
    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
    n += target.size(0)

    top1 = (acc1 / n) * 100
    top5 = (acc5 / n) * 100

    return top1, top5

def map_multilabel_list(output, target):
    """
    Computes the mean Average Precision (mAP) for multilabel classification.

    Args:
        output: Tensor of shape (N, C), where N is the number of samples,
                and C is the number of classes. Contains predicted scores.
        target: List of lists, where target[i] contains the correct labels
                for the i-th sample (variable number of labels per sample).

    Returns:
        mAP: Mean Average Precision across all classes.
    """
    num_classes = output.size(1)
    num_samples = len(target)

    # Flatten target into binary matrix for easier AP computation
    binary_target = torch.zeros((num_samples, num_classes), dtype=torch.int)
    for i, labels in enumerate(target):
        for label in labels:
            binary_target[i, label] = 1

    # Calculate AP for each class
    ap_per_class = []
    for c in range(num_classes):
        # Extract the predictions and true labels for class `c`
        preds = output[:, c]  # Shape: (N,)
        true_labels = binary_target[:, c]  # Shape: (N,)

        # Sort by prediction confidence
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_true_labels = true_labels[sorted_indices]

        # Compute Precision and Recall
        tp = sorted_true_labels.cumsum(dim=0)  # True Positives cumulative sum
        fp = (~sorted_true_labels.bool()).cumsum(dim=0)  # False Positives cumulative sum
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / sorted_true_labels.sum()  # Recall normalized by total positives

        # Adjust precision to match recall[1:] - recall[:-1]
        precision = precision[:-1]  # Remove last precision value to match ΔRecall size

        # Average Precision for this class
        ap = (precision * (recall[1:] - recall[:-1])).sum().item() if recall.numel() > 1 else 0.0
        ap_per_class.append(ap)

    # Mean Average Precision
    mAP = sum(ap_per_class) / len(ap_per_class)
    return mAP
