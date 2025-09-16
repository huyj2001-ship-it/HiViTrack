#!/bin/bash
# =========================================
# HiViTrack Environment Installation Script
# Python 3.7.16 + PyTorch 1.13.1 (CUDA 11.6)
# =========================================

# 1. Create and activate a virtual environment
echo ">>> Creating conda environment: hivitrack"
conda create -y -n hivitrack python=3.7.16
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hivitrack

# 2. install PyTorch + CUDA 11.6 
echo ">>> Installing PyTorch 1.13.1 with CUDA 11.6"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Install project dependencies
echo ">>> Installing project dependencies"
pip install \
    attributee==0.1.8 \
    bidict==0.22.1 \
    cachetools==5.3.2 \
    colorama==0.4.6 \
    dominate==2.9.1 \
    easydict==1.12 \
    einops==0.8.1 \
    gdown==5.2.0 \
    huggingface_hub==0.16.4 \
    ipython==7.33.0 \
    ipywidgets==8.1.7 \
    jpeg4py==0.1.4 \
    lazy_object_proxy==1.9.0 \
    lmdb==1.4.1 \
    matplotlib==3.5.3 \
    numba==0.56.4 \
    numpy==1.21.6 \
    opencv_python==4.9.0.80 \
    packaging==25.0 \
    pandas==1.3.5 \
    phx_class_registry==4.0.6 \
    Pillow==9.5.0 \
    pycocotools==2.0.7 \
    PyLaTeX==1.4.2 \
    PyYAML==6.0.2 \
    recommonmark==0.7.1 \
    requests==2.32.5 \
    scipy==1.7.3 \
    setuptools==65.6.3 \
    six==1.16.0 \
    tensorboardX==2.6.4 \
    thop==0.1.1.post2402281031 \
    tikzplotlib==0.10.1 \
    timm==0.6.13 \
    tqdm==4.64.1

# 4. Check the installation results
echo ">>> Checking installation..."
python -c "import sys, torch; print('Python:', sys.version); print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Available:', torch.cuda.is_available())"

echo ">>> Environment setup complete!"
