eval "$(conda shell.bash hook)"

git submodule update --init --recursive

conda create -n atlas python=3.7.5 tensorflow=1.15
conda activate atlas

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib tensorboard scipy scikit-image tqdm opencv-python imageio-ffmpeg gdown
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

cd layered-neural-atlases
cd thirdparty/RAFT/
./download_models.sh
cd ../..
