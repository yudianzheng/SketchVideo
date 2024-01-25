cd data
conda activate atlas
python process_dataset.py --mask_folder DAVIS/Annotations/Full-Resolution --image_folder DAVIS/JPEGImages/Full-Resolution --output_folder dataset --size 1080 1080

cd ..