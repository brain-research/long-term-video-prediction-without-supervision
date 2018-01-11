# Unsupervised Hierarchical Video Prediction

This is the implementation of
https://openreview.net/forum?id=rkmtTJZCb&noteId=rkmtTJZCb.

It can predict future frames in a video given the first few frames as context
and optionally the agent's action.

An encoder network infers a high level structure from a frame. A predictor
network predicts that structure into the future, and a VAN generates the
predicted frame from the predicted structure.

## Training

The network can be trained in different ways:

__E2E__: Train the network end to end to minimize the loss with the predicted
frame.

__EPEV__: Train the encoder and predictor together so the high level structure
is easy to predict. Also train the encoder and VAN together so the high level
structure is informative enough to generate the frame.

__Individual__: Train each network individually using a ground truth pose as the
high level structure.

__E2E with pose__: A hybrid of the E2E and individual methods.

## Commands

These commands have the best known hyperparameters for each mode and dataset.
The learning rate and batch size were optimized for parallel async training on
32 GPUs, so they may not be optimal for training on a single GPU.

For all commands, set "model_dir" and "event_log_dir" to the location where the
model and tensorboard logs should be saved. Set the data_dir to the directory
with your training data. Specify the pretrain_path flag if imgnet_pretrain is
set.

Use the same commands for validation except:

*   Add the nois_training flag and set run_mode to "eval".

*   Set data_pattern to "validation"

*   Remove the enc_keep_prob and van_keep_prob flags.

### Robot Push dataset

E2E:

`python prediction_train.py --model_mode e2e --dataset_type robot
--all_learning_rate 1e-05 --enc_size_set 16 --enc_keep_prob .65 --van_keep_prob
.9 --batch_size 8 --sequence_length 20 --skip_num 1 --run_mode "train"
--is_training --train_steps 3000000 --clip_gradient_norm .01`

EPEV:

`python prediction_train.py --model_mode epev --dataset_type robot
--imgnet_pretrain --all_learning_rate 1e-05 --enc_pred_loss_scale 1
--enc_pred_loss_scale_delay 6e5 --enc_size_set 32 --enc_keep_prob .65
--van_keep_prob .9 --batch_size 8 --sequence_length 20 --skip_num 1 --run_mode
"train" --is_training --train_steps 3000000 --clip_gradient_norm .01`

Individual:

`python prediction_train.py --model_mode individual --dataset_type robot
--enc_learning_rate 1e-5 --pred_learning_rate_map 3e-4 --van_learning_rate 3e-5
--enc_size_set 12 --enc_keep_prob .75 --van_keep_prob 1.0 --batch_size 8
--sequence_length 20 --skip_num 1 --run_mode "train" --is_training --train_steps
3000000 --clip_gradient_norm .01`

### Humans dataset

E2E:

`python prediction_train.py --model_mode e2e --dataset_type human
--all_learning_rate 1e-05 --enc_size_set 32 --enc_keep_prob .65 --van_keep_prob
.9 --batch_size 8 --sequence_length 64 --skip_num 2 --context_frames 5
--run_mode "train" --is_training --train_steps 3000000 --clip_gradient_norm .01`

EPEV:

`python prediction_train.py --model_mode epev --dataset_type human
--imgnet_pretrain --all_learning_rate 1e-05 --enc_pred_loss_scale .1
--enc_pred_loss_scale_delay 6e5 --enc_size_set 32 --enc_keep_prob .65
--van_keep_prob .9 --batch_size 8 --sequence_length 64 --skip_num 2
--context_frames 5 --run_mode "train" --is_training --train_steps 3000000
--clip_gradient_norm .01`
