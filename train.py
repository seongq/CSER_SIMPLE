import os
import pickle
from datetime import datetime
import pytz

import numpy as np, argparse, time, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import Model, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import torch.nn.functional as F
from utils import seed_everything,compute_detailed_metrics
from datetime import datetime
import pytz
# seed = 67137 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD


def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])



def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
    

def train_or_eval_graph_model(model,
                              loss_function, 
                              dataloader, 
                              epoch, 
                              cuda,
                              optimizer=None, 
                              train=False,
                              ):
    losses, preds, labels = [], [], []
    vids = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed_number)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-2]] if cuda else data[:-2]
        

        textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
        
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

    
        log_prob = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
    
    
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    vids += data[-1]
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=300, metavar='E', help='number of epochs')
    parser.add_argument('--graph_type', default='hyper', help='hyper/relation/GCN3/DeepGCN/MMGCN/MMGCN2')
    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')
    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')
    parser.add_argument('--mm_fusion_mthd', default='concat_DHT', help='method to use multimodal information: concat, gated, concat_subsequently')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test', choices = ("IEMOCAP", "MELD"))
    parser.add_argument('--num_graph_layers', type=int, default=4, help='num of GNN layers')
    parser.add_argument("--seed_number", type=int, default=1, required=True)
    parser.add_argument("--graph_masking", default=True, action="store_false")
    args = parser.parse_args()
    
    
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)
    timestamp_str = now_kst.strftime("%Y%m%d%H%M")
    print(timestamp_str)    
    print(args)
    
    name_ = '_'+args.modals+'_'+args.graph_type+'_'+args.Dataset
    
        
        
    print(name_)

    
    

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        device = torch.device("cuda:0")
        print('Running on GPU')
    else:
        device = torch.device("cpu")
        print('Running on CPU')


    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    seed_number = args.seed_number
    # feature dimension
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    D_m = 1024
    
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    
    
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1


    seed_everything(seed_number)
    print(n_speakers)
    model = Model(D_m, 
                  D_g, 
                  graph_h,
                  n_speakers=n_speakers,
                  n_classes=n_classes,
                  dropout=args.dropout,
                  alpha=args.alpha,
                  D_m_v = D_visual,
                  D_m_a = D_audio,
                  dataset=args.Dataset,
                  num_graph_layers = args.num_graph_layers,
                  graph_masking=args.graph_masking)

    name = 'Graph'

    if cuda:
        model.to(device)

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        #loss_function = FocalLoss()
        loss_function  = nn.NLLLoss(loss_weights.to(device) if cuda else loss_weights)


        

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr
    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)
    else:
        print("There is no such dataset")

    best_fscore, best_acc, best_loss, best_label_f1, best_label_acc, best_pred_f1, best_pred_acc , best_mask = -1000, -1000, None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []



    model_save_dir = os.path.join("./save_folder", args.Dataset, f"original___graphmasking_{args.graph_masking}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    
    pickle_path = os.path.join(f"./result/{args.Dataset}", "result.pkl")
    
    temporary_pickle_dir_path = os.path.dirname(pickle_path)
    os.makedirs(temporary_pickle_dir_path, exist_ok=True)
    
    
    best_f1_model_path = None
    best_acc_model_path = None
    for e in range(n_epochs):
        print(f"epoch: {e}")
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model,  
                                                                                 loss_function,
                                                                                 train_loader,
                                                                                 e, 
                                                                                 cuda,
                                                                                 optimizer,
                                                                                 True,
                                                                                 )
        
        
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model, 
                                                                              loss_function, 
                                                                              valid_loader,
                                                                              e, 
                                                                              cuda,                   
                                                                              )
        test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model,
                                                                                               loss_function,
                                                                                               test_loader,
                                                                                               e,
                                                                                               cuda
                                                                                               )
        all_fscore.append(test_fscore)
        all_acc.append(test_acc)

      
            
        best_mask = None
        f1_metrics = compute_detailed_metrics(test_label, test_pred, sample_weight=best_mask)
        wf1 = f1_score(test_label, test_pred, sample_weight=best_mask, average='weighted')
        acc = accuracy_score(test_label, test_pred)
        class_accuracy = f1_metrics["class_accuracy"]
        class_f1 = f1_metrics["class_f1"]
        weighted_accuracy = f1_metrics['weighted_accuracy']
        weighted_f1 = f1_metrics['weighted_f1']

        result_dictionary = {
            'epoch': e,
            'f1_w': weighted_f1,
            'acc_w': weighted_accuracy,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_fscore': train_fscore,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_fscore': test_fscore,
            'seed': seed_number
        }

        for i in range(len(class_accuracy)):
            result_dictionary[f"acc_{i}"] = class_accuracy[i]
        for i in range(len(class_f1)):
            result_dictionary[f"f1_{i}"] = class_f1[i]

        mode_str = f"ORIGINAL__graph_masking_{args.graph_masking}"
        result_dictionary["mode"] = mode_str

        # 모델 파일 이름
        filename = f"model_f1_{test_fscore:.2f}_acc_{acc*100:.2f}_epoch_{e}_seed_{seed_number}.pth"
        model_path = os.path.join(model_save_dir, filename)
        result_dictionary['path'] = model_path

        # 모델 저장
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": f1_metrics,
            "seed": seed_number,
        }, model_path)

        # 결과 누적 저장
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                tempdata = pickle.load(f)
            for key in result_dictionary:
                tempdata.setdefault(key, []).append(result_dictionary[key])
        else:
            tempdata = {key: [val] for key, val in result_dictionary.items()}

        with open(pickle_path, "wb") as f:
            pickle.dump(tempdata, f)
            
        elapsed_time = round(time.time() - start_time, 2)

        print(f"[Epoch {e+1:03d}]")
        print(f"  ▶ Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_fscore:.2f}")
        print(f"  ▶ Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%, F1: {valid_fscore:.2f}")
        print(f"  ▶ Test  Loss: {test_loss:.4f},  Acc: {test_acc:.2f}%, F1: {test_fscore:.2f}")
        print(f"  ▶ Weighted Accuracy: {weighted_accuracy:.2f}%, Weighted F1: {weighted_f1:.2f}")
        print(f"  ▶ Saved model to: {model_path}")
        print(f"  ⏱ Time elapsed: {elapsed_time} sec")
        print("-" * 60)
                    
    
