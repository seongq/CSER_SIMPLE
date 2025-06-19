import os
import pickle
from datetime import datetime
import pytz
import json
from pathlib import Path
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
    parser.add_argument('--epochs', type=int, default=999, metavar='E', help='number of epochs')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test', choices = ("IEMOCAP", "MELD"))
    parser.add_argument('--num_graph_layers', type=int, default=4, help='num of GNN layers')
    parser.add_argument("--seed_number", type=int, default=1, required=True)
    parser.add_argument("--graph_masking", default=True, action="store_false")
    
    parser.add_argument("--spk_embs", default='avt', choices= ("NO", 'a', 'v', 't', 'av', 'at', 'vt', 'avt'))
    parser.add_argument("--using_lstms", default="avt", choices= ("NO", 'a', 'v', 't', 'av', 'at', 'vt', 'avt'))
    parser.add_argument("--aligns", default="to_t", choices= ("NO", "to_a", "to_v", "to_t"))
    args = parser.parse_args()
    
    
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)
    timestamp_str = now_kst.strftime("%Y%m%d%H%M")
    print(args)
    
    main_name = "gnn_layers_"+str(args.num_graph_layers)+"_spk_embs_"+args.spk_embs+"_"+"using_lstms_"+args.using_lstms+"_"+"aligns_"+args.aligns+"_datasets_"+args.Dataset+"_"+"seed_"+str(args.seed_number)+f"_{timestamp_str}"
    
        
        
    print(main_name)

    
    

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
    # print(n_speakers)
    model = Model(D_m, 
                  D_g, 
                  graph_h,
                  n_speakers=n_speakers,
                  n_classes=n_classes,
                  dropout=args.dropout,
                  D_m_v = D_visual,
                  D_m_a = D_audio,
                  num_graph_layers = args.num_graph_layers,
                  graph_masking=args.graph_masking,
                  spk_embs=args.spk_embs,
                    using_lstms = args.using_lstms,
                    aligns =args.aligns                  
                  )


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




    model_save_dir = os.path.join("./save_folder", main_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    
    csv_path = os.path.join(f"./save_folder/{main_name}", "results.csv")
    
    temporary_csv_dir_path = os.path.dirname(csv_path)
    os.makedirs(temporary_csv_dir_path, exist_ok=True)
    
    
    import csv
    
    if not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline='') as f:
            writer = csv.writer(f)
            COLUMNS = []
            COLUMNS.append('epoch')
            
          
            COLUMNS.append("train_loss")
            COLUMNS.append("train_acc")
            COLUMNS.append("train_fscore")
            COLUMNS.append("test_loss")
            COLUMNS.append("test_acc")
            COLUMNS.append("test_fscore")
            
            for i in range(n_classes):
                COLUMNS.append(f'ACC_{i}')
                

            for i in range(n_classes):
                COLUMNS.append(f"F1_{i}")
                
            writer.writerow(COLUMNS)
    
    
    args_save_path = os.path.join(model_save_dir, "settings.json")
    
    # Namespace → dict 변환 후 저장
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    
    for e in range(n_epochs):
        epoch = str(e).zfill(3)
        print(f"epoch: {epoch} ")

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
          
            
        f1_metrics = compute_detailed_metrics(test_label, test_pred, sample_weight=None)
        class_accuracy = f1_metrics["class_accuracy"]
        
        class_f1 = f1_metrics["class_f1"]
        
        
       
        weighted_accuracy = f1_metrics['weighted_accuracy']
        weighted_f1 = f1_metrics['weighted_f1']

        filename = f"epoch_{epoch}.pth"
        model_path = os.path.join(model_save_dir, filename)
        
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            CONTENTS = [epoch, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore ]
            for i in range(n_classes): #ACC
                CONTENTS.append(round(class_accuracy[i], 2))
            for i in range(n_classes): #F1
                CONTENTS.append(round(class_f1[i], 2))
            writer.writerow(CONTENTS)

        # 모델 저장
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": f1_metrics,
            "seed": seed_number,
        }, model_path)

       
        elapsed_time = round(time.time() - start_time, 2)

        print(f"[Epoch {e+1:03d}]")
        print(f"  ▶ Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_fscore:.2f}")
        print(f"  ▶ Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%, F1: {valid_fscore:.2f}")
        print(f"  ▶ Test  Loss: {test_loss:.4f},  Acc: {test_acc:.2f}%, F1: {test_fscore:.2f}")
        print(f"  ▶ Weighted Accuracy: {weighted_accuracy:.2f}%, Weighted F1: {weighted_f1:.2f}")
        print(f"  ▶ Saved model to: {model_path}")
        print(f"  ⏱ Time elapsed: {elapsed_time} sec")
        print("-" * 60)
                    
    
