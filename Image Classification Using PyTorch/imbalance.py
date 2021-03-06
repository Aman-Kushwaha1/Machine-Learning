# EECS 545 Fall 2021
from dataset import DogCatDataset
from train import evaluate_loop, train
from transfer import load_pretrained

def per_class_accuracy(y_true, y_pred, num_classes=2):
    """
    Compute the per-class accuracy given true and predicted labels.
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - per_class_acc: per-class accuracy, list of floats
    """
    # TODO (part h): compute the per-class accuracy
    m = y_pred.size()[0]
    d=0
    dc = 0
    c=0
    cc = 0
    for i in range(m):
        if y_true[i] == 1:
            d += 1
            if y_pred[i] == y_true[i]:
                dc += 1
    else:
        c += 1
        if y_pred[i] == y_true[i]:
            cc += 1

    pcad = (dc/d)
    pcac = (cc/c)
    return [pcad, pcac]


def precision(y_true, y_pred):
    """
    Compute the precision given true and predicted labels. Treat the dog class (label=1) as the positive class.
    Precision = TP / (TP + FP)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - prec: precision, float
    """
    # TODO (part h): compute the precision
    m = y_pred.size()[0]
    d =0
    dc = 0
    for i in range(m):
        if y_pred[i] == 1:
            d += 1
            if y_pred[i] == y_true[i]:
                dc += 1
    out = dc/d
    return out


def recall(y_true, y_pred):
    """
    Compute the recall given true and predicted labels. Treat the dog class (label=1) as the positive class.
    Recall = TP / (TP + FN)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - rec: recall, float
    """
    # TODO (part h): compute the recall
    m = y_pred.size()[0]
    d =0
    dc = 0
    for i in range(m):
        if y_true[i] == 1:
            d += 1
            if y_pred[i] == y_true[i]:
                dc += 1
    out = dc/d
    return out

def f1_score(y_true, y_pred):
    """
    Compute the f1-score given true and predicted labels. Treat the dog class (label=1) as the positive class.
    F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    Inputs:
        - y_true: true labels, tensor with shape=(num_examples)
        - y_pred: predicted labels, tensor with shape=(num_examples)
    Returns:
        - f1: f1-score, float
    """
    # TODO (part h): compute the f1-score
    m = y_pred.size()[0]
    d =0
    dc = 0
    for i in range(m):
        if y_true[i] == 1:
            d += 1
            if y_pred[i] == y_true[i]:
                dc += 1
    recall1 = dc/d
    #calculating precision
    d =0
    dc = 0
    for i in range(m):
        if y_pred[i] == 1:
            d += 1
            if y_pred[i] == y_true[i]:
                dc += 1
    precision1 = dc/d
    out = (2*(precision1*recall1)/(precision1 + recall1))
    return out


def compute_metrics(dataset, model):
    y_true, y_pred, _ = evaluate_loop(dataset.val_loader, model)
    print('Per-class accuracy: ', per_class_accuracy(y_true, y_pred))
    print('Precision: ', precision(y_true, y_pred))
    print('Recall: ', recall(y_true, y_pred))
    print('F1-score: ', f1_score(y_true, y_pred))


if __name__ == '__main__':
    # model with normal cross-entropy loss
    config = {
        'dataset_path': 'data/images/dogs_vs_cats_imbalance',
        'batch_size': 4,
        # 'ckpt_force': True,
        'ckpt_path': 'checkpoints/imbalance',
        'plot_name': 'Imbalance',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
    }
    dataset = DogCatDataset(config['batch_size'], config['dataset_path'])
    model = load_pretrained(num_classes=2)
    train(config, dataset, model)
    compute_metrics(dataset, model)

    # model with weighted cross-entropy loss
    config = {
        'ckpt_path': 'checkpoints/imbalance_weighted',
        'plot_name': 'Imbalance-Weighted',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'use_weighted': True,
    }
    model_weighted = load_pretrained(num_classes=2)
    train(config, dataset, model_weighted)
    compute_metrics(dataset, model_weighted)
