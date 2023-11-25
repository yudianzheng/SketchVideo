eval "$(conda shell.bash hook)"

conda create -n clipavideo python=3.7.5 tensorflow=1.15
conda activate clipavideo

pip3 install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu116
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
