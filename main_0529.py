import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as auc3
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# from model_0511_full import Drugcell
from model_0529 import Drugcell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--node_dim', type=int, default=36)
    # parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    args = parser.parse_args()

    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    seed_torch()

    rna_seq = np.load('data/rna.npy', allow_pickle=True).astype(float)
    rna_seq = sklearn.preprocessing.scale(rna_seq, with_mean=True, with_std=True, axis=0)
    rna_seq = torch.as_tensor(rna_seq, dtype=torch.float32).to(device).transpose(0, 1)

    gene_cnv = np.load('data/cnv.npy', allow_pickle=True).astype(float)
    gene_cnv = torch.as_tensor(gene_cnv, dtype=torch.float32).to(device).transpose(0, 1)

    sensitive = np.load('data/sensitivity_index_list.npy', allow_pickle=True)
    resistant = np.load('data/resistant_index_list.npy', allow_pickle=True)

    np.random.seed(42)
    np.random.shuffle(sensitive)
    np.random.shuffle(resistant)

    def split_data(data):
        n = len(data)
        train = data[:int(n * 0.8)]
        val = data[int(n * 0.8):int(n * 0.9)]
        test = data[int(n * 0.9):]
        return train, val, test

    s_train, s_val, s_test = split_data(sensitive)
    r_train, r_val, r_test = split_data(resistant)

    train_index = np.vstack((s_train, r_train)).astype(int)
    val_index = np.vstack((s_val, r_val)).astype(int)
    test_index = np.vstack((s_test, r_test)).astype(int)

    train_target = torch.from_numpy(train_index[:, 2]).to(device)
    val_target = torch.from_numpy(val_index[:, 2]).to(device)
    test_target = torch.from_numpy(test_index[:, 2]).to(device)

    train_index = torch.from_numpy(train_index).to(device)
    val_index = torch.from_numpy(val_index).to(device)
    test_index = torch.from_numpy(test_index).to(device)

    drug_fea = np.load('drug_fea.npy', allow_pickle=True)
    drug_fea = [torch.as_tensor(df).to(device) for df in drug_fea]

    drug_adj = np.load('dict.npy', allow_pickle=True).item()
    drug_adj = {k: torch.from_numpy(v).to(device) for k, v in drug_adj.items()}

    drug_graphs = torch.load("data/drug_graphs.pt")

    model = Drugcell(
        num_tpm=rna_seq.shape[1],
        num_genecnv=gene_cnv.shape[1],
        node_dim=args.node_dim,
        drug_feat=75
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01)).to(device)

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    saved_once = False  # 用于记录是否已经保存过一次
    save_path = "save_model/best_model.pt"

    for epoch in range(args.epoch):
        print(f"Epoch: {epoch + 1:03d}")
        model.train()
        optimizer.zero_grad()

        loss, y, target = model(rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs, train_index, train_target, logit_scale)
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = (y.argmax(dim=1) == target).float().mean()
        auc = roc_auc_score(target.cpu().numpy(), y[:, 1].detach().cpu().numpy())
        precision, recall, _ = precision_recall_curve(target.cpu().numpy(), y[:, 1].detach().cpu().numpy())
        aupr = auc3(recall, precision)
        print(f"Train loss: {loss.item():.4f} | AUC: {auc:.4f} | AUPR: {aupr:.4f} | Acc: {acc.item():.4f}")

        model.eval()
        with torch.no_grad():
            val_loss, y_val, val_target = model(
                rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs, val_index, val_target, logit_scale
            )

            val_pred = y_val[:, 1].detach().cpu().numpy()
            val_true = val_target.cpu().numpy()

            val_auc = roc_auc_score(val_true, val_pred)
            precision, recall, _ = precision_recall_curve(val_true, val_pred)
            val_aupr = auc3(recall, precision)
            val_acc = (y_val.argmax(dim=1) == val_target).float().mean().item()

            print(f"Val loss: {val_loss.item():.4f} | AUC: {val_auc:.4f} | AUPR: {val_aupr:.4f} | Acc: {val_acc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print("✅ Best model updated.")


    # 载入最佳模型做测试
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        test_loss, y_test, test_target = model(rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs, test_index, test_target, logit_scale)
        acc = (y_test.argmax(dim=1) == test_target).float().mean()
        auc = roc_auc_score(test_target.cpu().numpy(), y_test[:, 1].cpu().numpy())
        precision, recall, _ = precision_recall_curve(test_target.cpu().numpy(), y_test[:, 1].cpu().numpy())
        aupr = auc3(recall, precision)

        print(f"\n🧪 Final Test Results:")
        print(f"AUC: {auc:.4f} | AUPR: {aupr:.4f} | Accuracy: {acc.item():.4f}")