# PAM-CDR

We additionally evaluated the performance of PAM-CDR on anticancer drug response datasets from the **Genomics of Drug Sensitivity in Cancer (GDSC)** dataset and the **Cancer Therapeutics Response Portal (CTRP)** dataset.

After data filtering, identifier matching, and feature alignment, the processed GDSC dataset contains **221 drugs**, **928 cell lines**, and **163,531 valid drug–cell-line pairs**, including **101,633 resistant samples** and **61,898 sensitive samples**， and the CTRP dataset contains **470 drugs**, **795 cell lines**, and **45,669 valid drug–cell-line pairs**, including **21,857 resistant samples** and **23,812 sensitive samples**..

Following a recent publication in **Nature Communications**: ***DrEval: Critical Evaluation of Drug Response Prediction Models with DrEval (Bernett et al., 2026)***, we adopted the **leave-cell-line-out (LCO)** and **leave-drug-out (LDO)** settings to evaluate PAM-CDR to unseen cell lines and unseen drugs, respectively. We additionally included a conventional **Original Setting** and a more stringent **Leave-Drug&Cell-Out** setting, in which both drugs and cell lines in the test set are unseen during training.

The data were divided into training, validation, and test sets using an 8:1:1 ratio. Comparative experiments were conducted under the following four evaluation settings:

* **Original Setting:** Drug–cell-line pairs are randomly divided into training, validation, and test sets.
* **LCO (Leave-Cell-Line-Out):**  Cell lines in the validation and test sets do not appear in the training set. This setting evaluates generalization to unseen cell lines.
* **LDO (Leave-Drug-Out):**  Drugs in the validation and test sets do not appear in the training set. This setting evaluates generalization to unseen drugs.
* **Leave-Drug&Cell-Out:** Both drugs and cell lines in the test set are unseen during training. The strict prefiltering procedure removes overlapping or invalid pairs before model training.

---

## Results on GDSC


### Original Setting

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.9130 |     0.8694 |     0.8368 |       0.7410 |            0.9063 |
| DeepTTA                            |     0.9032 |     0.8521 |     0.8430 |       0.7279 |            0.8966 |
| GraphCDR                           |     0.9161 |     0.8727 |     0.8384 |       0.7424 |            0.9113 |
| GraphDRP                           |     0.9310 |     0.8972 |     0.8553 |       0.7993 |            0.9300 |
| RefDNN                             |     0.8554 |     0.7799 |     0.7896 |       0.5426 |            0.8938 |
| **PAM-CDR (Proposed)**             | **0.9387** | **0.9089** | **0.8558** |   **0.8321** |        **0.9358** |

### Leave-Cell-Line-Out 

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.8829 |     0.8107 |     0.8059 |       0.5000 |            0.9032 |
| GraphCDR                           |     0.8592 |     0.7577 |     0.7752 |       0.5570 |            0.8983 |
| GraphDRP                           | **0.8978** | **0.8382** | **0.8123** |   **0.6415** |        **0.9089** |
| DeepTTA                            |     0.8796 |     0.8020 |     0.8038 |       0.6015 |            0.8945 |
| RefDNN                   |     0.6146 |     0.4837 |     0.4677 |       0.4991 |            0.6208 |
| **PAM-CDR (Proposed)**             |     0.8898 |     0.8194 |     0.8117 |       0.5893 |            0.9044 |

### Leave-Drug-Out 

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.6605 |     0.5208 |     0.6453 |       0.6748 |            0.5000 |
| DeepTTA                            |     0.6517 |     0.5136 |     0.5432 |       0.6708 |            0.3964 |
| GraphCDR                           |     0.6173 |     0.4595 |     0.6280 |       0.6658 |            0.5944 |
| GraphDRP                           |     0.6920 |     0.5513 |     0.6645 |       0.6488 |            0.6422 |
| RefDNN                             |     0.6325 |     0.4664 |     0.6400 |       0.5145 |            0.6339 |
| **PAM-CDR (Proposed)**             | **0.7660** | **0.6441** | **0.7108** |   **0.6752** |        **0.7575** |



### Leave-Drug&Cell-Out

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.5000 | **0.6828** |     0.6344 |       0.5000 |            0.5000 |
| DeepTTA                            |     0.5288 |     0.3813 |     0.6156 |       0.5984 |            0.5464 |
| GraphCDR                           |     0.4998 |     0.3530 |     0.3656 |       0.4073 |            0.5000 |
| GraphDRP                           |     0.5229 |     0.3762 |     0.5306 |   **0.6336** |            0.5205 |
| RefDNN                             |     0.6474 |     0.5602 |     0.6757 |       0.5541 |            0.6578 |
| **PAM-CDR (Proposed)**             | **0.6574** |     0.5653 | **0.6984** |       0.5842 |        **0.6612** |



---

## Results on CTRP

### Original Setting

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.9187 |     0.9208 |     0.8446 |       0.8986 |            0.7031 |
| NIHGCN                             | **0.9612** | **0.9666** | **0.8940** |   **0.9458** |        **0.9544** |
| DeepTTA                            |     0.8871 |     0.8714 |     0.8290 |       0.8778 |            0.6485 |
| GraphDRP                           |     0.9318 |     0.9394 |     0.8573 |       0.9037 |            0.7740 |
| **PAM-CDR (Proposed)**             |     0.9430 |     0.9490 |     0.8728 |       0.9202 |            0.8150 |

### Leave-Cell-Line-Out 

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.5609 |     0.6087 |     0.5109 |       0.5000 |            0.6906 |
| NIHGCN                             |     0.8258 |     0.8302 |     0.7589 |       0.7899 |        **0.7517** |
| GraphDRP                           |     0.6097 |     0.6456 |     0.5568 |       0.6337 |            0.6495 |
| DeepTTA                            |     0.7840 |     0.7849 |     0.6775 |       0.7661 |            0.6274 |
| **PAM-CDR (Proposed)**             | **0.8262** | **0.8427** | **0.7595** |   **0.7936** |            0.7433 |

### Leave-Drug-Out 

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.9022 |     0.9032 | **0.8281** |   **0.8617** |            0.5000 |
| NIHGCN                             |     0.8951 |     0.8960 |     0.8176 |       0.8507 |           **0.6093** |
| DeepTTA                            |     0.8841 |     0.8863 |     0.8147 |       0.8527 |            0.5569 |
| GraphDRP                           |     0.9022 |     0.9022 |     0.8139 |       0.8581 |            0.5779 |
| **PAM-CDR (Proposed)**             | **0.9028** | **0.9056** |     0.8279 |       0.8616 |            0.5631 |

### Leave-Drug&Cell-Out

| Model                              |        AUC |       AUPR |   Accuracy | Per-Drug AUC | Per-Cell-Line AUC |
| ---------------------------------- | ---------: | ---------: | ---------: | -----------: | ----------------: |
| LogReg (Drug + Cell-Line Identity) |     0.5000 |     0.7500 |     0.5000 |       0.5000 |            0.5000 |
| DeepTTA                            |     0.7554 | **0.8064** |     0.6900 |       0.7271 |            0.5152 |
| GraphDRP                           |     0.5350 |     0.4680 |     0.5815 |       0.5979 |            0.5579 |
| NIHGCN                             |     0.7706 |     0.7660 |     0.6923 |       0.7645 |        **0.6578** |
| **PAM-CDR (Proposed)**             | **0.7952** |     0.8059 | **0.7534** |   **0.8212** |            0.6288 |

---

## References

###  DrEval

> Bernett, J., Iversen, P., Picciani, M. et al. Critical evaluation of drug response prediction models with DrEval. Nat Commun 17, 4238 (2026). https://doi.org/10.1038/s41467-026-72903-w

### RefDNN

> Choi, Jonghwan, Sanghyun Park, and Jaegyoon Ahn. "RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance." Scientific reports 10.1 (2020): 1861.

### drGAT

> Inoue, Yoshitaka, et al. "drgat: Attention-guided gene assessment of drug response utilizing a drug-cell-gene heterogeneous network." (2024).