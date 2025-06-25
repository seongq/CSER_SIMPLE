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
from utils import seed_everything,compute_detailed_metrics, str2bool
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
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        if args.MRL: # MRL 실행
            if train: # train일 땐 logits 모아둔거랑, logit 나오니까
                # print("MRL traing 중")
                output_log_probs, log_prob = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
            else: # eval일 때 logit 하나만 나오지
                # print("MRL eval중")
                log_prob = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
            loss = loss_function(log_prob, label)
            
            if train: # 여러 logit에 대하여 loss 구해서 더해주기
                for MRL_log_prob in output_log_probs:
                    if args.MRL_loss_combination == "sum":
                        loss += loss_function(MRL_log_prob, label) # 합하는게 좋다.
                    elif args.MRL_loss_combination == "average":
                        loss = loss/(len(output_log_probs)+1)
            else: 
                pass
        elif args.MKD:
            if train:
                log_prob,log_prob_a, log_prob_v, log_prob_t, log_prob_a_teacher, log_prob_v_teacher, log_prob_t_teacher = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
                loss_a_teacher = loss_function(log_prob_a_teacher, label)
                loss_v_teacher = loss_function(log_prob_v_teacher, label)
                loss_t_teacher = loss_function(log_prob_t_teacher, label)
                
                log_prob_a_pseudo = log_prob_a_teacher.detach()
                log_prob_v_pseudo = log_prob_v_teacher.detach()
                log_prob_t_pseudo = log_prob_t_teacher.detach()
                
                loss_kd = F.kl_div(log_prob_a,log_prob_a_pseudo, log_target=True,reduction="batchmean") + F.kl_div( log_prob_v,log_prob_v_pseudo, log_target=True,reduction="batchmean")+F.kl_div(log_prob_t, log_prob_t_pseudo, log_target=True,reduction="batchmean")
                
                loss = loss_a_teacher + loss_v_teacher + loss_t_teacher + loss_kd
                
                if "a" in args.auxillary_classifier:
                    loss += loss_function(log_prob_a, label)
                if "t" in args.auxillary_classifier:
                    loss += loss_function(log_prob_t, label)
                if "v" in args.auxillary_classifier:
                    loss += loss_function(log_prob_v, label)
                
                loss = 0
            else:
                log_prob = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
                loss = 0
            loss += loss_function(log_prob, label)
        else:
            log_prob = model([textf1,textf2,textf3,textf4], qmask, lengths, acouf, visuf, epoch)
            loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        # print(preds)
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
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test', choices = ("IEMOCAP", "MELD"))
    parser.add_argument('--num_graph_layers', type=int, default=4, help='num of GNN layers')
    parser.add_argument("--seed_number", type=int, default=1)
    parser.add_argument("--graph_masking", default=True, type=str2bool)
    
    parser.add_argument("--spk_embs", default='avt', choices= ("NO", 'a', 'v', 't', 'av', 'at', 'vt', 'avt'))
    parser.add_argument("--using_lstms", default="avt", choices= ("NO", 'a', 'v', 't', 'av', 'at', 'vt', 'avt'))
    parser.add_argument("--aligns", default="to_t", choices= ("NO", "to_a", "to_v", "to_t"))
    parser.add_argument("--MRL", type=str2bool, default=False)
    parser.add_argument("--MRL_efficient", type=str2bool, default=False)
    parser.add_argument("--num_MRL_partition", type=int, default=0)
    parser.add_argument("--MRL_loss_combination", default="sum", choices=("NO", "sum", "average"))
    
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--mask_prob", default=0.5, type=float)
    parser.add_argument("--MKD", default=False,  type=str2bool)
    
    
    
    parser.add_argument("--num_graph_layers_a", default=4, type=int)
    parser.add_argument("--num_graph_layers_v", default=4, type=int)
    parser.add_argument("--num_graph_layers_t", default=4, type=int)
    
    parser.add_argument("--graph_masking_a", default=True, type=str2bool)
    parser.add_argument("--graph_masking_v", default=True, type=str2bool)
    parser.add_argument("--graph_masking_t", default=True, type=str2bool)
    
    parser.add_argument("--spk_embs_uni_modal_a", default=True, type=str2bool)
    parser.add_argument("--spk_embs_uni_modal_v", default=True, type=str2bool)
    parser.add_argument("--spk_embs_uni_modal_t", default=True, type=str2bool)
    
    parser.add_argument("--lstm_unimodal_a", type=str2bool, default=True)
    parser.add_argument("--lstm_unimodal_v", type=str2bool, default=True)
    parser.add_argument("--lstm_unimodal_t", type=str2bool, default=True)
    
    parser.add_argument("--aligns_uni_modal_a", type=str2bool, default=True)
    parser.add_argument("--aligns_uni_modal_v", type=str2bool, default=True)
    parser.add_argument("--aligns_uni_modal_t", type=str2bool, default=True)
    
    parser.add_argument("--num_heads_a", type=int, default=2)
    parser.add_argument("--num_heads_v", type=int, default=2)
    parser.add_argument("--num_heads_t", type=int, default=2)
    
    parser.add_argument("--mask_prob_a", type=float, default=0.5)
    parser.add_argument("--mask_prob_v", type=float, default=0.5)
    parser.add_argument("--mask_prob_t", type=float, default=0.5)
    
    parser.add_argument("--auxillary_classifier", type=str, default="avt", choices=("a", "v", "t", "av", "at", "vt", "avt"))
        
    
    parser.add_argument("--debug_mode", default=True, type=str2bool)
    
    args = parser.parse_args()
    if args.debug_mode:
        print("[INFO] debug mode:")
    else:
        print("[INFO] not debug mode, real implementation")
    assert not (not args.MRL and args.MRL_efficient), \
    "MRL must be True when using MRL_efficient."

    assert not (not args.MRL and args.num_MRL_partition > 0), \
        "num_MRL_partition must be 0 when MRL is False."

    assert not (args.MRL and args.num_MRL_partition == 0), \
        "num_MRL_partition must be > 0 when MRL is True."

    assert not (args.MRL and args.num_MRL_partition > 10), \
        "num_MRL_partition must be <= 10 when MRL is True."

    assert not (not args.MRL and args.MRL_loss_combination != "NO"), \
        "MRL_loss_combination must be 'NO' when MRL is False."

    assert not (args.MRL and args.MRL_loss_combination == "NO"), \
        "MRL_loss_combination must not be 'NO' when MRL is True."
        
    assert not (args.MRL==True and args.MKD == True)
        
    
    # timestamp
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)
    timestamp_str = now_kst.strftime("%Y%m%d%H%M")

    # 기본 이름 구성 요소
    parts = [
        f"mask_prob_{args.mask_prob}",
        f"num_heads_{args.num_heads}",
        f"gnn_layers_{args.num_graph_layers}",
        f"spk_embs_{args.spk_embs}",
        f"using_lstms_{args.using_lstms}",
        f"aligns_{args.aligns}",
        f"datasets_{args.Dataset}",
        f"seed_{args.seed_number}",
        f"MKD_{args.MKD}",
        f"timestamp_{timestamp_str}"
    ]

    # MRL 관련 설정 추가
    if args.MRL:
        parts.insert(0, f"partition_{args.num_MRL_partition}")
        parts.insert(0, f"MRLCOMB_{args.MRL_loss_combination}")
        prefix = "MRL_efficient" if args.MRL_efficient else "MRL"
        parts.insert(0, prefix)

    # 최종 이름 생성
    main_name = "_".join(parts)
    print(main_name)
    
        
        
    # print(main_name)

    
    

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
                    aligns =args.aligns,
                    MRL = args.MRL,
                    MRL_efficient = args.MRL_efficient,
                    num_MRL_partition = args.num_MRL_partition,
                    num_heads = args.num_heads,
                    mask_prob = args.mask_prob,
                    MKD = args.MKD,
                    num_graph_layers_a=args.num_graph_layers_a,
                 num_graph_layers_v=args.num_graph_layers_v,
                 num_graph_layers_t=args.num_graph_layers_t,
                 graph_masking_a=args.graph_masking_a,
                 graph_masking_v=args.graph_masking_v,
                 graph_masking_t=args.graph_masking_t,
                 spk_embs_uni_modal_a=args.spk_embs_uni_modal_a,
                 spk_embs_uni_modal_v=args.spk_embs_uni_modal_v,
                 spk_embs_uni_modal_t=args.spk_embs_uni_modal_t,
                 lstm_unimodal_a=args.lstm_unimodal_a,
                 lstm_unimodal_v=args.lstm_unimodal_v,
                 lstm_unimodal_t=args.lstm_unimodal_t,
                 aligns_uni_modal_a=args.aligns_uni_modal_a,
                 aligns_uni_modal_v=args.aligns_uni_modal_v,
                 aligns_uni_modal_t=args.aligns_uni_modal_t,
                 num_heads_a=args.num_heads_a,
                 num_heads_v=args.num_heads_v,
                 num_heads_t=args.num_heads_t,
                 mask_prob_a=args.mask_prob_a,
                 mask_prob_v=args.mask_prob_v,
                 mask_prob_t=args.mask_prob_t
                  )

# parser.add_argument("--MRL", type=str2bool, default=False)
#     parser.add_argument("--MRL_efficient", type=str2bool, default=False)
#     parser.add_argument("--num_MRL_partition", type=int, default=3)
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
    if args.debug_mode:
        pass
    else:
        os.makedirs(model_save_dir, exist_ok=True)
    
    
    csv_path = os.path.join(f"./save_folder/{main_name}", "results.csv")
    
    temporary_csv_dir_path = os.path.dirname(csv_path)
    
    if args.debug_mode:
        pass
    else:
        os.makedirs(temporary_csv_dir_path, exist_ok=True)
    
    
    import csv
    if not args.debug_mode and not os.path.isfile(csv_path):
        with open(csv_path, mode="w", newline='') as f:
            writer = csv.writer(f)

            COLUMNS = ['main_name']  # 가장 앞 열
            COLUMNS += ['epoch', 'train_loss', 'train_acc', 'train_fscore',
                        'test_loss', 'test_acc', 'test_fscore']

            for i in range(n_classes):
                COLUMNS.append(f'ACC_{i}')
            for i in range(n_classes):
                COLUMNS.append(f'F1_{i}')

            args_keys_sorted = sorted(vars(args).keys())
            COLUMNS += [f"arg_{k}" for k in args_keys_sorted]

            writer.writerow(COLUMNS)

# ===== JSON으로 args 저장 =====
    if not args.debug_mode:
        args_save_path = os.path.join(model_save_dir, "settings.json")
        with open(args_save_path, "w") as f:
            json.dump(vars(args), f, indent=4)

# ===== TRAINING LOOP =====
    for e in range(n_epochs):
        print(main_name)
        epoch = str(e).zfill(3)
        print(f"epoch: {epoch}")

        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(
            model, loss_function, train_loader, e, cuda, optimizer, train=True
        )

        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(
            model, loss_function, valid_loader, e, cuda
        )

        test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(
            model, loss_function, test_loader, e, cuda
        )

        f1_metrics = compute_detailed_metrics(test_label, test_pred)
        class_accuracy = f1_metrics["class_accuracy"]
        class_f1 = f1_metrics["class_f1"]

        # 저장 경로
        filename = f"epoch_{epoch}.pth"
        model_path = os.path.join(model_save_dir, filename)

        # ===== CSV 내용 저장 =====
        if not args.debug_mode:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                CONTENTS = [
                    main_name,  # ← 여기 변경됨
                    epoch,
                    train_loss, train_acc, train_fscore,
                    test_loss, test_acc, test_fscore
                ]
                CONTENTS += [round(class_accuracy[i], 4) for i in range(n_classes)]
                CONTENTS += [round(class_f1[i], 4) for i in range(n_classes)]

                args_dict = vars(args)
                CONTENTS += [args_dict[k] for k in sorted(args_dict.keys())]

                writer.writerow(CONTENTS)
        
        # if args.debug_mode:
        #     pass
        
        # else:
        #     if not os.path.isfile(csv_path):
        #         with open(csv_path, mode="w", newline='') as f:
        #             writer = csv.writer(f)
        #             COLUMNS = []
        #             COLUMNS.append('epoch')
                    
                
        #             COLUMNS.append("train_loss")
        #             COLUMNS.append("train_acc")
        #             COLUMNS.append("train_fscore")
        #             COLUMNS.append("test_loss")
        #             COLUMNS.append("test_acc")
        #             COLUMNS.append("test_fscore")
                    
        #             for i in range(n_classes):
        #                 COLUMNS.append(f'ACC_{i}')
                        

        #             for i in range(n_classes):
        #                 COLUMNS.append(f"F1_{i}")
                        
        #             writer.writerow(COLUMNS)
        
        
        # args_save_path = os.path.join(model_save_dir, "settings.json")
        
        # # Namespace → dict 변환 후 저장
        
        # if args.debug_mode:
        #     pass
        # else:
                
        #     with open(args_save_path, "w") as f:
        #         json.dump(vars(args), f, indent=4)
        
        
        # for e in range(n_epochs):
        #     print(main_name)
        #     epoch = str(e).zfill(3)
        #     print(f"epoch: {epoch} ")

        #     start_time = time.time()

        #     train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model,  
        #                                                                              loss_function,
        #                                                                              train_loader,
        #                                                                              e, 
        #                                                                              cuda,
        #                                                                              optimizer,
        #                                                                              True,
        #                                                                              )
            
            
        #     valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model, 
        #                                                                           loss_function, 
        #                                                                           valid_loader,
        #                                                                           e, 
        #                                                                           cuda,                   
        #                                                                           )
        #     test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model,
        #                                                                                            loss_function,
        #                                                                                            test_loader,
        #                                                                                            e,
        #                                                                                            cuda
        #                                                                                            )
            
                
        #     f1_metrics = compute_detailed_metrics(test_label, test_pred, sample_weight=None)
        #     class_accuracy = f1_metrics["class_accuracy"]
            
        #     class_f1 = f1_metrics["class_f1"]
            
            
        
        weighted_accuracy = f1_metrics['weighted_accuracy']
        weighted_f1 = f1_metrics['weighted_f1']

        #     filename = f"epoch_{epoch}.pth"
        #     model_path = os.path.join(model_save_dir, filename)
            
            
            
        #     if args.debug_mode:
        #         pass
            
        #     else:
        #         with open(csv_path, mode='a', newline='') as f:
        #             writer = csv.writer(f)
        #             CONTENTS = [epoch, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore ]
        #             for i in range(n_classes): #ACC
        #                 CONTENTS.append(round(class_accuracy[i], 2))
        #             for i in range(n_classes): #F1
        #                 CONTENTS.append(round(class_f1[i], 2))
        #             writer.writerow(CONTENTS)


        if args.debug_mode:
            pass
        else:
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
        if not args.debug_mode:
            print(f"  ▶ Saved model to: {model_path}")
        else:
            print(f" Debug mode, models are not saved.")
        print(f"  ⏱ Time elapsed: {elapsed_time} sec")
        print("-" * 60)
        # print(f"  ⚙️ Model settings summary:")
        # for k, v in vars(args).items():
        #     print(f"     - {k}: {v}")
        print("-" * 60)

                    
    
