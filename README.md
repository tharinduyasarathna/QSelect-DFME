# QSelect-DFME: Query-Efficient Data-Free Model Extraction for Robustness Validation of DL-Based AAD in SDN-IoT

Official implementation of **QSelect-DFME**, a query-efficient data-free model extraction framework for evaluating the robustness of deep learning-based autonomous anomaly detection (AAD) systems in **SDN-IoT environments**.

---

## 📌 Overview

Deep learning-based AAD systems are deployed as **black-box services** in SDN-IoT networks. Evaluating their robustness under **realistic constraints** (no data access, limited queries) remains challenging.

This work:
- Reformulates **Data-Free Model Extraction (DFME)** as a **query-constrained robustness evaluation problem**
- Proposes **QSelect-DFME**, a framework for efficient behavioral extraction
- Focuses on **ranking-based behavior**, aligned with real-world anomaly detection decisions

---

## 🚀 Key Features

- Query-efficient extraction (≤ 2k queries)  
- Manifold-guided filtering (PCA-based)  
- Diversity-aware query selection (k-center)  
- Ranking-aligned optimization  
- Designed for black-box SDN-IoT settings  

---

## 🏗️ Method Pipeline

1. Generate a synthetic candidate pool  
2. Apply manifold filtering  
3. Select diverse queries  
4. Query target model  
5. Train surrogate model  

---

## 📊 Datasets

- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)  
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  
- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)  
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)  
- [InSDN](https://github.com/CanadianInstituteForCybersecurity/InSDN)  
- [ASEADOS-SDN-IoT](https://aseados.ucd.ie/datasets/SDN-IoT/)  

> Download datasets and place them inside the `data/` directory.

---

## 🤖 Models

- Autoencoder (AE) *(implemented via PyOD)*  
- Variational Autoencoder (VAE) *(implemented via PyOD)*  
- Deep SVDD *(implemented via PyOD)*  
- DROCC  
- NeuTraL-AD   

---

## ▶️ Run

```bash
python3 -m experiments.qselect_dfme_score_binary
```

### Baselines

```bash
python3 -m experiments.tabextractor_score_binary
python3 -m experiments.tempest_score_binary
```

---

## ⚖️ License

MIT License
