# cd diffvg && python setup.py install

cd CLIPavideo

python run_object_sketching.py --target_file $1 --focus foreground --data_folder ../data/dataset/$1 --consist_param 3.0 --num_strokes 50 --num_iter 3001 --atlas_epoch '050000' --clip_model_name "RN101" --num_of_frames 60
python run_object_sketching.py --target_file $1 --focus background --data_folder ../data/dataset/$1 --consist_param 0.15 --num_strokes 256 --num_iter 3001 --atlas_epoch '050000' --clip_model_name "ViT-B/32" --num_of_frames 60
