import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as auc3
from model import Drugcell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def safe_metrics(y_true, y_score, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float((y_pred == y_true).mean()) if len(y_true) > 0 else np.nan
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "aupr": np.nan, "acc": acc}

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(auc3(recall, precision)),
        "acc": acc,
    }

def split_80_10_10(arr, seed=42):

    arr = np.asarray(arr).copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)
    n = len(arr)
    train = arr[: int(n * 0.8)]
    val = arr[int(n * 0.8): int(n * 0.9)]
    test = arr[int(n * 0.9):]
    return train, val, test

def make_leave_drug_cell_out(pairs, seed=42):

    pairs = np.asarray(pairs).astype(int)
    cells = np.unique(pairs[:, 0])
    drugs = np.unique(pairs[:, 1])
    train_cells, val_cells, test_cells = split_80_10_10(cells, seed=seed)
    train_drugs, val_drugs, test_drugs = split_80_10_10(drugs, seed=seed)

    train_mask = np.isin(pairs[:, 0], train_cells) & np.isin(pairs[:, 1], train_drugs)
    val_mask = np.isin(pairs[:, 0], val_cells) & np.isin(pairs[:, 1], val_drugs)
    test_mask = np.isin(pairs[:, 0], test_cells) & np.isin(pairs[:, 1], test_drugs)

    used_mask = train_mask | val_mask | test_mask
    train = pairs[train_mask].astype(int)
    val = pairs[val_mask].astype(int)
    test = pairs[test_mask].astype(int)
    unused = pairs[~used_mask].astype(int)
    return train, val, test, unused


def label_count(arr):
    if len(arr) == 0:
        return {0: 0, 1: 0}
    vc = pd.Series(arr[:, 2].astype(int)).value_counts().to_dict()
    return {0: int(vc.get(0, 0)), 1: int(vc.get(1, 0))}

def to_df(arr, split_name):
    df = pd.DataFrame(arr.astype(int), columns=["cell_line_index", "drug_index", "label"])
    df["split"] = split_name
    return df

def per_group_auc(test_df, y_score, out_dir, group_col, group_name):
    out_dir = Path(out_dir)
    df = test_df.copy()
    df["score_sensitive"] = y_score
    auc_col = f"per_{group_name}_auc"
    rows = []
    for group_idx, g in df.groupby(group_col):
        y = g["label"].values.astype(int)
        s = g["score_sensitive"].values.astype(float)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        auc_val = np.nan if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))
        rows.append({
            group_col: int(group_idx),
            "n": int(len(g)),
            "n_sensitive": n_pos,
            "n_resistant": n_neg,
            auc_col: auc_val,
        })
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"test_per_{group_name}_auc.csv", index=False, encoding="utf-8-sig")
    valid_df = res.dropna(subset=[auc_col])
    macro = float(valid_df[auc_col].mean()) if len(valid_df) else np.nan
    weighted = float(np.average(valid_df[auc_col], weights=valid_df["n"])) if len(valid_df) else np.nan
    return macro, weighted, res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--node_dim", type=int, default=36)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--early_stop_patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="CTRP/")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--print_test_max_rows", type=int, default=50)
    args = parser.parse_args()


    if args.out_dir is None:
        args.out_dir = f"split_debug_leave_drug_cell_out_seed{args.seed}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path("save_model").mkdir(parents=True, exist_ok=True)
    save_path = f"save_model/best_model_newdata_leave_drug_cell_out_seed{args.seed}.pt"


    rna_seq = np.load(os.path.join(args.data_dir, "rna.npy"), allow_pickle=True).astype(float)
    print("RNA raw shape gene x cell:", rna_seq.shape)
    rna_seq = sklearn.preprocessing.scale(rna_seq, with_mean=True, with_std=True, axis=0)
    rna_seq = torch.as_tensor(rna_seq, dtype=torch.float32).to(device)

    gene_cnv = np.load(os.path.join(args.data_dir, "cnv.npy"), allow_pickle=True).astype(float)
    print("CNV raw shape gene x cell:", gene_cnv.shape)
    gene_cnv = torch.as_tensor(gene_cnv, dtype=torch.float32).to(device)

    sensitive = np.load(os.path.join(args.data_dir, "sensitive_filter.npy"),
                        allow_pickle=True).astype(int)
    resistant = np.load(os.path.join(args.data_dir, "resistant_filter.npy"),
                        allow_pickle=True).astype(int)

    sensitive[:, [0, 1]] = sensitive[:, [1, 0]]
    resistant[:, [0, 1]] = resistant[:, [1, 0]]

    sensitive[:, 2] = 1
    resistant[:, 2] = 0

    all_pairs_raw = np.vstack((sensitive, resistant)).astype(int)
    print("Raw pairs:", all_pairs_raw.shape)
    print("Raw label counts:", pd.Series(all_pairs_raw[:, 2]).value_counts().to_dict())


    train_index_np, val_index_np, test_index_np, unused_index_np = make_leave_drug_cell_out(all_pairs_raw, seed=args.seed)
    test_df = to_df(test_index_np, "test")

    if len(train_index_np) == 0 or len(val_index_np) == 0 or len(test_index_np) == 0:
        raise RuntimeError("One of train/val/test is empty. Choose another seed.")

    train_target = torch.from_numpy(train_index_np[:, 2]).long().to(device)
    val_target_tensor = torch.from_numpy(val_index_np[:, 2]).long().to(device)
    test_target_tensor = torch.from_numpy(test_index_np[:, 2]).long().to(device)

    train_index = torch.from_numpy(train_index_np).long().to(device)
    val_index = torch.from_numpy(val_index_np).long().to(device)
    test_index = torch.from_numpy(test_index_np).long().to(device)


    drug_fea = np.load(os.path.join(args.data_dir, "drug_fea.npy"), allow_pickle=True)
    drug_fea = [torch.as_tensor(df).to(device) for df in drug_fea]

    drug_adj = np.load(os.path.join(args.data_dir, "dict.npy"), allow_pickle=True).item()
    drug_adj = {k: torch.from_numpy(v).to(device) for k, v in drug_adj.items()}

    drug_graphs = torch.load(os.path.join(args.data_dir, "drug_graphs.pt"),weights_only=False)


    model = Drugcell(
        num_tpm=rna_seq.shape[1],
        num_genecnv=gene_cnv.shape[1],
        node_dim=args.node_dim,
        drug_feat=75,
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01)).to(device)

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    has_saved_checkpoint = False

    for epoch in range(args.epoch):
        print(f"Epoch: {epoch + 1:03d}")
        start_time = time.time()

        model.train()
        optimizer.zero_grad()
        loss, y, target = model(
            rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
            train_index, train_target, logit_scale,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_time = time.time() - start_time
        train_prob = y[:, 1].detach().cpu().numpy()
        train_true = target.detach().cpu().numpy()
        train_pred = y.argmax(dim=1).detach().cpu().numpy()
        train_metrics = safe_metrics(train_true, train_prob, train_pred)
        print(f"train Time: {train_time:.2f}s")
        print(
            f"Train loss: {loss.item():.4f} | AUC: {train_metrics['auc']:.4f} | "
            f"AUPR: {train_metrics['aupr']:.4f} | Acc: {train_metrics['acc']:.4f}"
        )

        val_start_time = time.time()
        model.eval()
        with torch.no_grad():
            val_loss, y_val, val_target_out = model(
                rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
                val_index, val_target_tensor, logit_scale,
            )
        val_time = time.time() - val_start_time

        val_prob = y_val[:, 1].detach().cpu().numpy()
        val_true = val_target_out.detach().cpu().numpy()
        val_pred = y_val.argmax(dim=1).detach().cpu().numpy()
        val_metrics = safe_metrics(val_true, val_prob, val_pred)
        print(f"val Time: {val_time:.2f}s")
        print(
            f"Val loss: {val_loss.item():.4f} | AUC: {val_metrics['auc']:.4f} | "
            f"AUPR: {val_metrics['aupr']:.4f} | Acc: {val_metrics['acc']:.4f}"
        )

        current_val_auc = val_metrics["auc"]
        if np.isnan(current_val_auc):
            current_val_auc = -np.inf

        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            has_saved_checkpoint = True
            print(f" Best model updated: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(
                    f" Early stopping at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch + 1} with AUC {best_val_auc:.4f}"
                )
                break

    if not has_saved_checkpoint:
        print(" No valid validation-AUC checkpoint was saved. Saving current model before test.")
        torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path,weights_only=False))
    # model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        test_start_time = time.time()
        test_loss, y_test, test_target_out = model(
            rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
            test_index, test_target_tensor, logit_scale,
        )
        test_time = time.time() - test_start_time

    test_prob = y_test[:, 1].detach().cpu().numpy()
    test_true = test_target_out.detach().cpu().numpy()
    test_pred = y_test.argmax(dim=1).detach().cpu().numpy()
    test_metrics = safe_metrics(test_true, test_prob, test_pred)

    print(f"test Time: {test_time:.2f}s")
    print("\n Final Test Results:")
    print(
        f"AUC: {test_metrics['auc']:.4f} | AUPR: {test_metrics['aupr']:.4f} | "
        f"Accuracy: {test_metrics['acc']:.4f} | Test Time: {test_time:.2f}s"
    )

    pred_df = test_df.copy()
    pred_df["score_sensitive"] = test_prob
    pred_df["pred_label"] = test_pred
    pred_df["true_label"] = test_true
    pred_df["correct"] = pred_df["pred_label"].astype(int) == pred_df["true_label"].astype(int)
    pred_df.to_csv(Path(args.out_dir) / "test_predictions_index.csv", index=False, encoding="utf-8-sig")


    macro_pdrug_auc, weighted_pdrug_auc, per_drug_df = per_group_auc(
        test_df, test_prob, args.out_dir, group_col="drug_index", group_name="drug"
    )
    macro_pcell_auc, weighted_pcell_auc, per_cell_df = per_group_auc(
        test_df, test_prob, args.out_dir, group_col="cell_line_index", group_name="cell"
    )

    print(f"Per-drug AUC macro on nonconstant test drugs: {macro_pdrug_auc:.4f}")
    print(f"Per-cell AUC macro on nonconstant test cells: {macro_pcell_auc:.4f}")