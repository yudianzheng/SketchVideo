cd layered-neural-atlases

python preprocess_optical_flow.py --vid-path ../data/dataset/$1/imgs_crop --max_long_edge 224
# python preprocess_optical_flow.py --vid-path ../data/dataset/$1/imgs_bg_crop --max_long_edge 224

python ../data/args.py --data_folder ../data/dataset/$1/imgs_crop \
 --bg_folder ../data/dataset/$1/imgs_bg_crop \
 --evaluate_at 10000 --result_folder ../data/dataset/$1/results --resx 224 --resy 224 
 
python train.py config/config.json
