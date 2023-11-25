cd E2FGVI

# python test.py --model e2fgvi_hq --video DAVIS/Annotations/Full-Resolution --mask /apdcephfs_cq2/share_1290939/shadowcun/sketching-video/$1/masks \
 --ckpt release_model/E2FGVI-HQ-CVPR22.pth --set_size --output_folder /apdcephfs_cq2/share_1290939/shadowcun/sketching-video/$1 --width 960 --height 540

cd ../data
python crop.py --image_folder /apdcephfs_cq2/share_1290939/shadowcun/sketching-video/$1 --mask_folder /apdcephfs_cq2/share_1290939/shadowcun/sketching-video/$1/masks --size 1080 1080
