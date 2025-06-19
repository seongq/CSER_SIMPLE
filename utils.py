import numpy as np, argparse, time, random
import torch



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed=10):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def compute_detailed_metrics(labels, preds, sample_weight=None):
    report = classification_report(labels, preds, sample_weight=sample_weight, output_dict=True)
    cm = confusion_matrix(labels, preds, sample_weight=sample_weight)

    # class-wise accuracy
    class_acc = (cm.diagonal() / cm.sum(axis=1) * 100 ).tolist()
    # print(class_acc)
    # class-wise F1
    class_f1 = [report[str(i)]['f1-score'] * 100 for i in range(len(class_acc))]

    # weighted average accuracy = accuracy_score with sample_weight
    weighted_accuracy = accuracy_score(labels, preds, sample_weight=sample_weight) * 100

    # weighted average F1
    weighted_f1 = f1_score(labels, preds, average='weighted', sample_weight=sample_weight) * 100

    return {
        'class_accuracy': class_acc,                 # 리스트: 각 클래스별 accuracy
        'class_f1': class_f1,                        # 리스트: 각 클래스별 F1
        'weighted_accuracy': weighted_accuracy,      # float
        'weighted_f1': weighted_f1                   # float
    }