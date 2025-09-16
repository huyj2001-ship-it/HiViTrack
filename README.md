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

- Python 3.7.16  
- PyTorch 1.13.1+cu116  
- CUDA 11.6  
- GPU: RTX 3090 (24GB) or equivalent (training tested on 2× RTX 3090)  
