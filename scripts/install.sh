eval "$(conda shell.bash hook)"

conda create -n atlas python=3.7.5 tensorflow=1.15
conda activate atlas

conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install matplotlib tensorboard scipy scikit-image tqdm opencv-python imageio-ffmpeg gdown icecream
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

cd layered-neural-atlases
git submodule update --init --recursive
cd thirdparty/RAFT/
./download_models.sh
cd ../../..
conda deactivate


conda create -n clipavideo python=3.7.5 tensorflow=1.15 -y
conda activate clipavideo

pip3 install -y torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu116
pip install natsort
cd diffvg
git submodule update --init --recursive
conda install -y numpy
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
cd ..

cd CLIPavideo
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install ipywidgets
pip install icecream
cd ..