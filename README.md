# HiViTrack

Official PyTorch implementation of **HiViTrack**, a Hierarchical Vision Transformer with Efficient Target-Prompt Update for Visual Object Tracking.  

![示例图片](./HiViTrack/tracking/OnlineHiViTr - V22.pdf)

---

## 🔑 Highlights

- **Hierarchical Vision Transformer Backbone** – Combines lightweight MLP-based early layers with Transformer-based deeper layers.  
- **SSDR (Shallow Spatial Details Retention)** – Preserves fine-grained spatial cues at low computational cost.  
- **DSMI (Deep Semantic Mutual Integration)** – Enhances discrimination via hierarchical template–search feature fusion.  
- **TPU (Target-Prompt Update)** – Efficient online template refinement during inference without heavy optimization.  
- Achieves superior performance on **LaSOT, TrackingNet, GOT-10k, VOT2022, LaSOText, and VastTrack** benchmarks.  

| Backbone | Tracker | Publication | LaSOT<br>AUC (%) | TrackingNet<br>AUC (%) | GOT-10k<br>AO (%) | GOT-10k<br>SR<sub>0.75</sub> (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| SiamRPN++ [9] | CVPR2019 | 49.6 | 73.3 | 51.7 | 32.5 | 34.0 |
| DiMP [13] | ICCV2019 | 56.9 | 74.0 | 61.1 | 49.2 | 39.2 |
| STARK [7] | ICCV2021 | 67.1 | 82.0 | 68.8 | 64.1 | - |
| OSTrack-256 [5] | ECCV2022 | 69.1 | 83.1 | 71.0 | 68.2 | 47.4 |
| ARTrack [31] | CVPR2023 | 70.4 | 84.2 | 73.5 | 70.9 | - |
| LoReTrack [33] | IROS2025 | 70.3 | 82.9 | 73.5 | 70.4 | 51.3 |
| SwinTrack-T [8] | NeurIPS2022 | 67.2 | 81.1 | 71.3 | 64.5 | 49.1 |
| MixFormer-1K [4] | CVPR2022 | 67.9 | 82.6 | 73.2 | 70.2 | - |
| SuperSBT-Base [37] | TPAMI2025 | 70.0 | 84.0 | 74.4 | 71.3 | 48.1 |
| **Offline HiViTrack** | **Ours** | 68.8 | 83.2 | 71.7 | 70.5 | 49.2 |
| **HiViTrack** | **Ours** | **70.3** | **84.5** | **73.0** | **71.3** | **51.1** |

---

## 📦 Installation

### Use the Anaconda
```
conda create -n Hivitrack python=3.7.16
conda activate Hivitrack
bash install.sh
```
### Put the tracking datasets in ./data. It should look like:
```
${HiViTrack_ROOT}
 -- data
     -- lasot
         |-- airplane
         |-- basketball
         |-- bear
         ...
     -- got10k
         |-- test
         |-- train
         |-- val
     -- coco
         |-- annotations
         |-- train2017
     -- trackingnet
         |-- TRAIN_0
         |-- TRAIN_1
         ...
         |-- TRAIN_11
         |-- TEST
```

### Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Train HiViTrack
```
bash tracking/train_hivitrack.sh
```

### Test and evaluate HiViTrack on benchmarks
```
bash tracking/test_hivitrack.sh
```
