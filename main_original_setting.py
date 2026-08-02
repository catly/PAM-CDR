import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc as auc3
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.preprocessing import scale
from model import Drugcell


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def metrics(y_true, y_score, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float((y_pred == y_true).mean())
    if np.unique(y_true).size < 2:
        return np.nan, np.nan, acc

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_value = float(roc_auc_score(y_true, y_score))
    aupr_value = float(auc3(recall, precision))
    return auc_value, aupr_value, acc


def per_group_auc(index_array, y_true, y_score, group_column):
    auc_values, group_sizes = [], []
    total_groups = 0

    for group_id in np.unique(index_array[:, group_column]):
        total_groups += 1
        mask = index_array[:, group_column] == group_id
        group_y = y_true[mask]

        if np.unique(group_y).size < 2:
            continue

        auc_values.append(roc_auc_score(group_y, y_score[mask]))
        group_sizes.append(mask.sum())

    valid_groups = len(auc_values)
    skipped_groups = total_groups - valid_groups

    if valid_groups == 0:
        return np.nan, np.nan, valid_groups, skipped_groups

    auc_values = np.asarray(auc_values, dtype=float)
    group_sizes = np.asarray(group_sizes, dtype=float)
    macro_auc = float(auc_values.mean())
    weighted_auc = float(np.average(auc_values, weights=group_sizes))
    return macro_auc, weighted_auc, valid_groups, skipped_groups


def split_data(data):
    n = len(data)
    return (
        data[: int(n * 0.8)],
        data[int(n * 0.8): int(n * 0.9)],
        data[int(n * 0.9):],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--node_dim", type=int, default=36)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--early_stop_patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
    )
    args = parser.parse_args()

    rna_seq = np.load(os.path.join(args.data_dir, "rna.npy"), allow_pickle=True).astype(float)
    rna_seq = scale(rna_seq, with_mean=True, with_std=True, axis=0)
    rna_seq = torch.as_tensor(rna_seq, dtype=torch.float32, device=device).transpose(0, 1)
    gene_cnv = np.load(os.path.join(args.data_dir, "cnv.npy"), allow_pickle=True).astype(float)
    gene_cnv = torch.as_tensor(gene_cnv, dtype=torch.float32, device=device).transpose(0, 1)

    sensitive = np.load(os.path.join(args.data_dir, "sensitivity_index_list.npy"), allow_pickle=True).astype(int)
    resistant = np.load(os.path.join(args.data_dir, "resistant_index_list.npy"), allow_pickle=True).astype(int)

    sensitive[:, 2] = 1
    resistant[:, 2] = 0
    rng = np.random.RandomState(args.seed)
    rng.shuffle(sensitive)
    rng.shuffle(resistant)

    s_train, s_val, s_test = split_data(sensitive)
    r_train, r_val, r_test = split_data(resistant)

    train_index_np = np.vstack((s_train, r_train)).astype(int)
    val_index_np = np.vstack((s_val, r_val)).astype(int)
    test_index_np = np.vstack((s_test, r_test)).astype(int)

    if min(len(train_index_np), len(val_index_np), len(test_index_np)) == 0:
        raise RuntimeError("Train, validation, or test set is empty.")

    train_index = torch.as_tensor(
        train_index_np, dtype=torch.long, device=device
    )
    val_index = torch.as_tensor(
        val_index_np, dtype=torch.long, device=device
    )
    test_index = torch.as_tensor(
        test_index_np, dtype=torch.long, device=device
    )

    train_target = train_index[:, 2]
    val_target = val_index[:, 2]
    test_target = test_index[:, 2]

    drug_fea = np.load(
        os.path.join(args.data_dir, "drug_fea.npy"), allow_pickle=True
    )
    drug_fea = [torch.as_tensor(x, device=device) for x in drug_fea]

    drug_adj = np.load(
        os.path.join(args.data_dir, "dict.npy"), allow_pickle=True
    ).item()
    drug_adj = {
        key: torch.as_tensor(value, device=device)
        for key, value in drug_adj.items()
    }
    drug_graphs = torch.load(
        os.path.join(args.data_dir, "drug_graphs.pt"), map_location=device
    )

    model = Drugcell(
        num_tpm=rna_seq.shape[1],
        num_genecnv=gene_cnv.shape[1],
        node_dim=args.node_dim,
        drug_feat=75,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch, eta_min=1e-6
    )
    logit_scale = nn.Parameter(
        torch.ones([], device=device) * np.log(1 / 0.01)
    )

    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None

    # Training
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()

        train_loss, _, _ = model(
            rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
            train_index, train_target, logit_scale,
        )
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss, val_output, val_target_out = model(
                rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
                val_index, val_target, logit_scale,
            )

        val_true = val_target_out.detach().cpu().numpy()
        val_score = val_output[:, 1].detach().cpu().numpy()
        val_pred = val_output.argmax(dim=1).detach().cpu().numpy()
        val_auc, val_aupr, val_acc = metrics(
            val_true, val_score, val_pred
        )

        print(
            f"Epoch {epoch + 1:04d} | Train loss: {train_loss.item():.4f} | "
            f"Val loss: {val_loss.item():.4f} | AUC: {val_auc:.4f} | "
            f"AUPR: {val_aupr:.4f} | Acc: {val_acc:.4f}"
        )

        current_val_auc = -np.inf if np.isnan(val_auc) else val_auc
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch: {best_epoch}, val AUC: {best_val_auc:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was obtained.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_loss, test_output, test_target_out = model(
            rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs,
            test_index, test_target, logit_scale,
        )

    test_true = test_target_out.detach().cpu().numpy()
    test_score = test_output[:, 1].detach().cpu().numpy()
    test_pred = test_output.argmax(dim=1).detach().cpu().numpy()
    test_auc, test_aupr, test_acc = metrics(
        test_true, test_score, test_pred
    )

    drug_macro, drug_weighted, valid_drugs, skipped_drugs = per_group_auc(
        test_index_np, test_true, test_score, group_column=1
    )
    cell_macro, cell_weighted, valid_cells, skipped_cells = per_group_auc(
        test_index_np, test_true, test_score, group_column=0
    )

    print("\nFinal Test Results")
    print(f"Best epoch: {best_epoch} | Best val AUC: {best_val_auc:.4f}")
    print(
        f"Test loss: {test_loss.item():.4f} | AUC: {test_auc:.4f} | "
        f"AUPR: {test_aupr:.4f} | Accuracy: {test_acc:.4f}"
    )
    print(f"Per-drug AUC: macro={drug_macro:.4f}")
    print(f"Per-cell AUC: macro={cell_macro:.4f}")
