# HiViTrack

Official PyTorch implementation of **HiViTrack**, a Hierarchical Vision Transformer with Efficient Target-Prompt Update for Visual Object Tracking.  

---

## 🔑 Highlights

- **Hierarchical Vision Transformer Backbone** – Combines lightweight MLP-based early layers with Transformer-based deeper layers.  
- **SSDR (Shallow Spatial Details Retention)** – Preserves fine-grained spatial cues at low computational cost.  
- **DSMI (Deep Semantic Mutual Integration)** – Enhances discrimination via hierarchical template–search feature fusion.  
- **TPU (Target-Prompt Update)** – Efficient online template refinement during inference without heavy optimization.  
- Achieves superior performance on **LaSOT, TrackingNet, GOT-10k, VOT2022, LaSOText, and VastTrack** benchmarks.  

---

## 📦 Installation

-Use the Anaconda
'''
conda create -n Hivitrack python=3.7.16
conda activate Hivitrack
bash install.sh
'''
