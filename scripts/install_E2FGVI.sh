conda env create -f E2FGVI/environment.yml
conda activate e2fgvi

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
