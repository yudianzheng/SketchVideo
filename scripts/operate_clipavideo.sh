eval "$(conda shell.bash hook)"

conda activate clipavideo

cd CLIPavideo
pwd

# python run_object_sketching.py --target_file $1 --focus foreground --data_folder ../data/dataset/$1 --consist_param 3.0 --frames_param 0.005 --num_strokes 55 --num_iter 1501 --atlas_epoch '010000' --clip_model_name "ViT-B/32" --num_of_frames 49 --clip_conv_layer_weights "0.0,1.0,1.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0" --clip_fc_loss_weight "0.1" --width "1.5" --device 1
# python run_object_sketching.py --target_file $1 --focus foreground --data_folder ../data/dataset/$1 --consist_param 2.0 --frames_param 0.005 --num_strokes 64 --num_iter 7501 --atlas_epoch '010000' --clip_model_name "RN101" --num_of_frames 49 --clip_RN_layer_weights "0.0,0.0,1.0,1.0,0.0" --clip_fc_loss_weight "0.1" --width "1.5" --device 1
python run_object_sketching.py --target_file $1 --focus foreground --data_folder ../data/dataset/$1 --consist_param 3.0 --frames_param 0.005 --num_strokes 64 --num_iter 1201 --atlas_epoch '010000' --clip_model_name "RN101" --num_of_frames 49 --clip_RN_layer_weights "0.0,0.0,1.0,1.0,0.0" --clip_fc_loss_weight "0.1" --width "1.5" --device 0


cd ..