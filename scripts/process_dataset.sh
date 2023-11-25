
# cd data

# python process_dataset.py --mask_folder /apdcephfs_cq2/share_1290939/shadowcun/datasets/DAVIS/Annotations/Full-Resolution \
#     --image_folder /apdcephfs_cq2/share_1290939/shadowcun/datasets/DAVIS/JPEGImages/Full-Resolution \
#     --output_folder /apdcephfs_cq2/share_1290939/shadowcun/sketching-video --size 1080 1080


# conda activate atlas
cd data
python process_dataset.py --mask_folder DAVIS/Annotations/Full-Resolution --image_folder DAVIS/JPEGImages/Full-Resolution --output_folder dataset --size 1080 1080
