import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from torch_geometric.data import Data
import torch
from tqdm import tqdm

from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from transformers import BertTokenizer

from utils_graph import mol_to_graph_data_obj_simple
from wrapper import preprocess_item


def cal_descriptors(smiles):
    # 将SMILES字符串转换为分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES representation.")
        return None
    # 计算分子描述符
    descriptors = {}
    # 获取10(15)个最常用的分子描述符
    descriptor_names = [
        'MolWt', 'MolLogP', 'NumRotatableBonds', 'TPSA', 'NumHDonors',
        'NumHAcceptors', 'RingCount', 'NumAromaticRings', 'FractionCSP3', 'BalabanJ'
    ]

    for name in descriptor_names:
        try:
            value = getattr(Descriptors, name)(mol)
            descriptors[name] = f"{value:.1f}"
        except Exception as e:
            print(f"Error calculating {name} descriptor: {e}")
            descriptors[name] = 0.0

    # 将描述符保存为文本格式
    prompt_text = ', '.join([f"{key}: {value}" for key, value in descriptors.items()])

    # 保存文本到文件
    # with open(output_file, 'w') as file:
    #     file.write(text_output)

    return prompt_text



def tokenizer_prompt(prompt_text):
    tokenizer = BertTokenizer.from_pretrained('../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
    sentence_token = tokenizer(text=prompt_text,
                                truncation=True,
                                padding='max_length',
                                add_special_tokens=False,
                                max_length=32,
                                return_tensors='pt',
                                return_attention_mask=True)
    input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
    attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
    return input_ids, attention_mask


def smiles2morgan(smiles):
    try:
        # 将 SMILES 转化为分子对象
        mol = Chem.MolFromSmiles(smiles)
        # 计算摩根指纹（ECFP4），2048 位
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        features = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, features)
    except:
        print('rdkit not found this smiles for morgan: ' + smiles + ' convert to all 0 features')
        features = np.zeros((2048,), dtype=int)
    return features

# 读取 drug 数据
df = pd.read_csv("../data/drugandsmile.csv")

# 遍历每个 SMILES
graphs = []
failed_drugs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    drug = row['drug']
    smiles = row['smiles']

    # mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.MolFromSmiles(smiles)

    if mol is None:
        print(f"[失败] 无法解析 SMILES：{drug} - {smiles}")
        failed_drugs.append(drug)
        continue

    try:
        graph = mol_to_graph_data_obj_simple(mol)
        graph = preprocess_item(graph)
        # graph.drug_name = drug  # 可选：给图加上名字属性

        # 添加prompt text
        prompt = cal_descriptors(smiles)
        drug_morgan = smiles2morgan(smiles)
        prompt_text, prompt_mask = tokenizer_prompt(prompt)
        graph.prompt_text = prompt_text
        graph.prompt_mask = prompt_mask
        graph.drug_morgan = drug_morgan
        graphs.append(graph)
    except Exception as e:
        print(f"[错误] 处理 {drug} 出错: {e}")
        failed_drugs.append(drug)

# 保存为 .pt 文件（包含所有图对象的列表）
torch.save(graphs, "drug_graphs.pt")
print(f"已保存 {len(graphs)} 个图结构到 drug_graphs.pt")
if failed_drugs:
    print(f"以下分子解析失败：{failed_drugs}")