#   --ckpt_path ./checkpoints/ \
#   --sequence_len 60 \
#   --traj_per_sample 50 \


python train_on_real_data.py \
  --batch_size 1 \
  --num_steps 15000 \
  --ckpt_path ./ \
  --model_name cotracker_three \
  --save_freq 200 \
  --sequence_len 80 \
  --eval_datasets none \
  --traj_per_sample 384 \
  --save_every_n_epoch 15 \
  --evaluate_every_n_epoch 15 \
  --model_stride 4 \
  --num_nodes 1 \
  --real_data_splits 0 \
  --num_virtual_tracks 64 \
  --mixed_precision \
  --random_frame_rate \
  --restore_ckpt ./checkpoints/baseline_offline.pth \
  --lr 0.00005 \
  --real_data_filter_sift \
  --validate_at_start \
  --offline_model \
  --limit_samples 1000 \
  --video_dir /l/users/malak.mansour/DEP/co-tracker/tif_attempts/my_videos \
  --annotation_dir /l/users/malak.mansour/DEP/co-tracker/tif_attempts/my_annotations