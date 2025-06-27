import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import GraphormerModel
from transformers import BertModel, BertConfig

from torch.utils.data import Dataset, DataLoader
from collator_CDR import collator
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Drugcell(nn.Module):
    def __init__(self, num_tpm, num_genecnv, node_dim, drug_feat):
        super(Drugcell, self).__init__()
        self.num_tpm = num_tpm
        self.num_genecnv = num_genecnv
        self.node_dim = node_dim
        self.drug_feat = drug_feat

        self.mlp_1 = MLP_1(num_tpm, node_dim)
        self.mlp_2 = MLP_2(num_genecnv, node_dim)

        self.alpha = nn.Parameter(torch.tensor(0.7))  # 设置初始值
        self.mlp_end = MLP_end(144, 2)  # 36 72 36
        self.weight = nn.Parameter(torch.Tensor(drug_feat, node_dim * 8)).to(device)  # GCN
        self.weight2 = nn.Parameter(torch.Tensor(node_dim * 8, node_dim * 2)).to(device)  # GCN
        self.prompt_proj = nn.Linear(768, drug_feat)
        self.reset_parameters()

        self.loss = nn.CrossEntropyLoss()

        self.prompt_encoder = PromptEncoder(pretrained=True)
        prompt_ckpt = torch.load('all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in prompt_ckpt:
            pretrained_dict = {"main_model." + k[20:]: v for k, v in prompt_ckpt.items()}
        elif 'bert.embeddings.word_embeddings.weight' in prompt_ckpt:
            pretrained_dict = {"main_model." + k[5:]: v for k, v in prompt_ckpt.items()}
        else:
            pretrained_dict = {"main_model." + k[12:]: v for k, v in prompt_ckpt.items()}
        self.prompt_encoder.load_state_dict(pretrained_dict, strict=False)

        drug_out_dim = 36
        self.drug_embedding = DrugEmbedding(drug_out_dim)

        self.cell_cross_attn = Cell_CrossAttention(dim1=36, dim2=36, embed_dim=64, num_heads=4, out_dim=36)
        self.drug_cross_attn = Drug_CrossAttention(dim1=72, dim2=36, embed_dim=64, num_heads=4, out_dim=108)
        self.GCNencoder = GCNEncoder(in_dim=75, hidden_dim1=64, hidden_dim2=32, out_dim=75)
        self.graph_prompt_CrossEncoder = Graph_prompt_CrossEncoder(text_dim=768, graph_dim=72)
        self.fp_prompt_CrossEncoder = FP_prompt_CrossEncoder(text_dim=768, fp_dim=36, n_heads=3)

    def norm(self, H, add=False):
        H = H.t()
        if not add:
            H = H * (torch.eye(H.shape[0]) == 0).float()
        else:
            H = H * (torch.eye(H.shape[0]) == 0).float().to(device) + torch.eye(H.shape[0]).float().to(device)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1).to(device)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).float().to(device)
        H = torch.mm(deg_inv, H)
        return H.t()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)


    def gcn_conv(self, X, H):

        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        X_1 = torch.mm(H.t(), X)
        X_2 = torch.mm(X_1, self.weight2)
        return torch.mm(H.t(), X_2)

    def compute_tpm(self, counts, lengths):
        counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        lengths = torch.nan_to_num(lengths, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(lengths, (int, float)):
            lengths = torch.full((counts.shape[1],), lengths, device=counts.device)

        counts_per_kb = counts / lengths
        total_counts = torch.sum(counts_per_kb, dim=1, keepdim=True)
        tpm = counts_per_kb / total_counts * 1e6
        log_tpm = torch.log2(tpm + 1)
        return torch.nan_to_num(log_tpm, nan=0.0, posinf=0.0, neginf=0.0)

    def gaussian_normalization(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-5)

    def forward(self, rna_seq, gene_cnv, drug_fea, drug_adj, drug_graphs, index_list, target, logit_scale):
        lengths = torch.full((rna_seq.shape[1],), 24358.0).cuda()
        rna_seq = self.compute_tpm(rna_seq, lengths)
        rna_seq1 = self.mlp_1(rna_seq)

        gene_cnv = self.compute_tpm(gene_cnv, lengths)
        gene_cnv1 = self.mlp_2(gene_cnv)

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = 1.0 - alpha

        rna_seq1 = alpha * rna_seq1
        gene_cnv1 = beta * gene_cnv1
        cell_fea = self.cell_cross_attn(rna_seq1, gene_cnv1)

        temp = 0
        for i in range(len(drug_fea)):

            prompt_text = drug_graphs[i].prompt_text
            prompt_mask = drug_graphs[i].prompt_mask
            prompt_text = prompt_text.to(device)
            prompt_mask = prompt_mask.to(device)

            prompt_emb, prompt_extended_mask = self.prompt_encoder(prompt_text, prompt_mask)
            # gcn_fea = F.relu(self.gcn_conv(drug_fea[i].to(torch.float32), drug_adj[i].to(torch.float32)))
            gcn_fea = self.GCNencoder(drug_fea[i].to(torch.float32), drug_adj[i].to(torch.float32))

            gcn_fea = self.graph_prompt_CrossEncoder(gcn_fea, prompt_emb)
            gcn_fea = torch.unsqueeze(torch.max(gcn_fea, dim=0)[0], dim=-1)   # max pooling

            # LiuC: fingerprint
            drug_morgan = drug_graphs[i].drug_morgan
            drug_morgan= torch.tensor(drug_morgan, dtype=torch.float32)
            drug_morgan = drug_morgan.to(device)
            fp_fea = self.drug_embedding(drug_morgan)
            fp_fea = fp_fea.unsqueeze(0)
            fp_fea = self.fp_prompt_CrossEncoder(fp_fea, prompt_emb)

            if temp == 0:
                drugfea_graph = gcn_fea
                drugfea_fp = fp_fea
            else:
                drugfea_graph = torch.cat((drugfea_graph, gcn_fea), dim=1)
                drugfea_fp = torch.cat((drugfea_fp, fp_fea), dim=0)
            temp = 1

        drugfea_graph = drugfea_graph.T
        # drug_fea = torch.cat((drugfea_graph, drugfea_fp), dim=1)
        drug_fea = self.drug_cross_attn(drugfea_graph, drugfea_fp)
        # drug_fea = drugfea_graph

        B_fea = torch.cat((cell_fea[index_list[:, 0], :], drug_fea[index_list[:, 1], :].float()), 1)
        B = self.mlp_end(B_fea)
        B = torch.squeeze(B)
        loss1 = self.loss(B, target)
        # return loss1, B, target, drug_fea
        # return loss1, B, target, B_fea
        return loss1, B, target, drugfea_graph
        # return loss1, B, target


class MLP_1(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.linear(x)
        if torch.isnan(out).any():
            print("⚠️ MLP_1 输出出现 NaN")
        return out


class MLP_2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.linear(x)
        if torch.isnan(out).any():
            print("⚠️ MLP_2 输出出现 NaN")
        return out


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super(GCNEncoder, self).__init__()
        self.gcn1 = nn.Linear(in_dim, hidden_dim1)
        self.gcn2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gcn3 = nn.Linear(hidden_dim2, out_dim)

    def normalize_adj(self, A, add_self_loop=True):
        if add_self_loop:
            A = A + torch.eye(A.size(0), device=A.device)
        D = torch.diag(torch.pow(A.sum(1), -0.5))
        return D @ A @ D

    def forward(self, X, A):
        A_hat = self.normalize_adj(A)

        # 第1层 GCN
        X = torch.matmul(A_hat, X)
        X = self.gcn1(X)
        X = F.relu(X)

        # 第2层 GCN
        X = torch.matmul(A_hat, X)
        X = self.gcn2(X)
        X = F.relu(X)

        # 第3层 GCN
        X = torch.matmul(A_hat, X)
        X = self.gcn3(X)

        return X  # 返回所有节点的最终表示


class Graph_prompt_CrossEncoder(nn.Module):
    def __init__(self, text_dim=768, graph_dim=72, n_heads=3):
        super().__init__()
        assert graph_dim % n_heads == 0, "graph_dim 必须能被 num_heads 整除"

        # 文本降维到图的维度
        self.text_proj = nn.Linear(text_dim, graph_dim)

        # 跨模态注意力：图为 Query，文本为 Key/Value（都用 72 维）
        self.cross_attn = nn.MultiheadAttention(embed_dim=graph_dim, num_heads=n_heads, batch_first=True)

        # 可选：融合模块（残差 + FFN）
        self.norm = nn.LayerNorm(graph_dim)
        self.ffn = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.ReLU(),
            nn.Linear(graph_dim, graph_dim)
        )

    def forward(self, gcn_fea, prompt_emb):
        """
        gcn_fea: [N_nodes, 75]             ← 图结构（不变）
        prompt_emb: [1, N_text, 768]       ← 文本模态输出
        """
        # 1. 文本降维到 72
        prompt_proj = self.text_proj(prompt_emb.squeeze(0))  # [N_text, 72]

        # 2. 图为 Query，文本为 Key/Value
        attn_output, _ = self.cross_attn(
            query=gcn_fea.unsqueeze(0),  # [1, N_nodes, 72]
            key=prompt_proj.unsqueeze(0),  # [1, N_text, 72]
            value=prompt_proj.unsqueeze(0)  # [1, N_text, 72]
        )  # → [1, N_nodes, 72]

        # 3. 残差 + FFN（可选）
        x = self.norm(attn_output + gcn_fea.unsqueeze(0))
        out = self.ffn(x)  # [1, N_nodes, 72]

        return out.squeeze(0)



class FP_prompt_CrossEncoder(nn.Module):
    def __init__(self, text_dim=768, fp_dim=36, n_heads=3):
        super().__init__()
        assert fp_dim % n_heads == 0, "fp_dim 必须能被 num_heads 整除"

        # 文本降维到指纹维度
        self.text_proj = nn.Linear(text_dim, fp_dim)

        # 跨模态注意力：FP 为 Query，文本为 Key/Value（都用 fp_dim 维）
        self.cross_attn = nn.MultiheadAttention(embed_dim=fp_dim, num_heads=n_heads, batch_first=True)

        # 可选：融合模块（残差 + FFN）
        self.norm = nn.LayerNorm(fp_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fp_dim, fp_dim),
            nn.ReLU(),
            nn.Linear(fp_dim, fp_dim)
        )

    def forward(self, fp_fea, prompt_emb):
        """
        fp_fea: [1, 36]                    ← 分子指纹特征（单个）
        prompt_emb: [1, N_text, 768]       ← 文本模态输出
        """
        # 保证 fp_fea 是 [1, 1, 36]
        if fp_fea.dim() == 2:
            fp_fea = fp_fea.unsqueeze(1)

        # 文本降维到 fp_dim
        prompt_proj = self.text_proj(prompt_emb.squeeze(0))

        # FP 为 Query，文本为 Key/Value
        attn_output, _ = self.cross_attn(
            query=fp_fea,
            key=prompt_proj.unsqueeze(0),
            value=prompt_proj.unsqueeze(0)
        )  # → [1, 1, 36]

        # 残差 + FFN
        x = self.norm(attn_output + fp_fea)
        out = self.ffn(x)

        return out.squeeze(1)




class Cell_CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, embed_dim=64, num_heads=4, out_dim=36):
        super().__init__()
        # feat1 → feat2
        self.q_proj_1 = nn.Linear(dim1, embed_dim)
        self.k_proj_1 = nn.Linear(dim2, embed_dim)
        self.v_proj_1 = nn.Linear(dim2, embed_dim)

        # feat2 → feat1
        self.q_proj_2 = nn.Linear(dim2, embed_dim)
        self.k_proj_2 = nn.Linear(dim1, embed_dim)
        self.v_proj_2 = nn.Linear(dim1, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 投影两种模态以做残差连接
        self.res_proj1 = nn.Linear(dim1, out_dim)
        self.res_proj2 = nn.Linear(dim2, out_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: Tensor [B, dim1] — e.g. RNA expression
            feat2: Tensor [B, dim2] — e.g. CNV features

        Returns:
            fused representation: Tensor [B, out_dim]
        """
        # feat1 attends to feat2
        q1 = self.q_proj_1(feat1).unsqueeze(1)  # [B, 1, E]
        k1 = self.k_proj_1(feat2).unsqueeze(1)
        v1 = self.v_proj_1(feat2).unsqueeze(1)
        out1, _ = self.attn(q1, k1, v1)         # [B, 1, E]

        # feat2 attends to feat1
        q2 = self.q_proj_2(feat2).unsqueeze(1)
        k2 = self.k_proj_2(feat1).unsqueeze(1)
        v2 = self.v_proj_2(feat1).unsqueeze(1)
        out2, _ = self.attn(q2, k2, v2)

        # concat and fuse
        fused = torch.cat([out1, out2], dim=-1).squeeze(1)  # [B, 2E] → [B, out_dim]
        fused = self.fusion(fused)

        # residual from both
        residual = 0.5 * (self.res_proj1(feat1) + self.res_proj2(feat2))
        return self.output_proj(fused + residual)



class Drug_CrossAttention(nn.Module):
    def __init__(self, dim1=72, dim2=36, embed_dim=64, num_heads=4, out_dim=108):
        super().__init__()
        # graph → fp
        self.query_proj_1 = nn.Linear(dim1, embed_dim)
        self.key_proj_1 = nn.Linear(dim2, embed_dim)
        self.value_proj_1 = nn.Linear(dim2, embed_dim)

        # fp → graph
        self.query_proj_2 = nn.Linear(dim2, embed_dim)
        self.key_proj_2 = nn.Linear(dim1, embed_dim)
        self.value_proj_2 = nn.Linear(dim1, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 🔧 投影两个模态用于残差融合
        self.graph_proj = nn.Linear(dim1, out_dim)
        self.fp_proj = nn.Linear(dim2, out_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, drugfea_graph, drugfea_fp):
        """
        drugfea_graph: [B, 72]
        drugfea_fp:    [B, 36]
        output:        [B, out_dim]
        """
        # graph → fp
        q1 = self.query_proj_1(drugfea_graph).unsqueeze(1)
        k1 = self.key_proj_1(drugfea_fp).unsqueeze(1)
        v1 = self.value_proj_1(drugfea_fp).unsqueeze(1)
        out1, _ = self.attn(q1, k1, v1)

        # fp → graph
        q2 = self.query_proj_2(drugfea_fp).unsqueeze(1)
        k2 = self.key_proj_2(drugfea_graph).unsqueeze(1)
        v2 = self.value_proj_2(drugfea_graph).unsqueeze(1)
        out2, _ = self.attn(q2, k2, v2)

        # 拼接注意力输出并融合
        fused = self.fusion(torch.cat([out1, out2], dim=-1).squeeze(1))

        # 残差连接（维度一致后融合）
        residual = 0.5 * (self.graph_proj(drugfea_graph) + self.fp_proj(drugfea_fp))
        out = self.output_proj(fused + residual)

        return out


class MLP_end(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_end, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.linear(x)


class DrugEmbedding(nn.Module):
    def __init__(self, out_dim):
        super(DrugEmbedding, self).__init__()
        # 仅使用 Morgan 指纹
        self.morgan_fp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, morgan_fp_input):
        # 输入是单个 Tensor，不再是元组
        return self.morgan_fp(morgan_fp_input)


class PromptEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(PromptEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = BertModel.from_pretrained(
                        'all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')
        # prompt_embedding
        self.embed = self.main_model.embeddings
        self.dropout = nn.Dropout(0.1)

        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        # device = input_ids.device
        device = torch.device(f'cuda:0')

        self.main_model = self.main_model.to(device)


        typ = torch.zeros(input_ids.shape).long().to(device)

        prompt_emb = self.embed(input_ids, token_type_ids=typ)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        prompt_extended_mask = (1.0 - extended_attention_mask) * -10000.0

        return prompt_emb, prompt_extended_mask
