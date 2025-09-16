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

| Backbone | Tracker | LaSOT<br>AUC (%) | TrackingNet<br>AUC (%) | GOT-10k<br>AO (%) | LaSOT<sub>ext</sub><br>AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| DiMP [13] | 56.9 | 74.0 | 61.1 | 39.2 |
| ARTrack [31] | 70.4 | 84.2 | 73.5 | - |
| LoReTrack [33] | 70.3 | 82.9 | 73.5 | 51.3 |
| SuperSBT-Base [37] | 70.0 | 84.0 | 74.4 | 48.1 |
| **Offline HiViTrack (Ours)** | 68.8 | 83.2 | 71.7 | 49.2 |
| **HiViTrack (Ours)** | **70.3** | **84.5** | **73.0** | **51.1** |

---

## 📦 Installation

-Use the Anaconda
```
conda create -n Hivitrack python=3.7.16
conda activate Hivitrack
bash install.sh
```


