# Hierarchical Long-term Video Prediction without Supervision

This is the implementation of [Hierarchical Long-term Video Prediction without Supervision](http://web.eecs.umich.edu/~honglak/icml2018-unsupHierarchicalVideoPred.pdf), to be published in ICML 2018.

It can predict future frames in a video given the first few frames as context
and optionally the agent's action.

An encoder network infers a high level structure from a frame. A predictor
network predicts that structure into the future, and a VAN generates the
predicted frame from the predicted structure.

The included code works for both the Humans 3.6M[^1] and the Robot Push
Dataset[^2], but the results are more impressive on the Humans dataset.

This code is tested on TensorFlow Version 1.7.0

## Training

The network can be trained in different ways:

__EPVA__: Train the encoder and predictor together so the high level structure
is easy to predict. Also train the encoder and VAN together so the high level
structure is informative enough to generate the frame.

__EPVA GAN__: Same as EPVA, but using an adversarial loss between the encoder
and predictor.

__E2E__: Train the network end to end to minimize the loss with the predicted
frame.

__Individual__: Train each network individually using a ground truth pose as the
high level structure.

__E2E with pose__: A hybrid of the E2E and individual methods.

## Get data

### Humans 3.6M

#### Use data on Google Cloud

By default the commands below will use the data stored on Google Cloud. This is
make it easier to start training since you won't need to download the data. To
use this option, follow these steps:

1.  Follow [these](https://cloud.google.com/sdk/downloads) instructions to
    install gcloud.
2.  Run `gcloud auth application-default login` from the command line. It
    doesn't matter what google account you sign in with.

#### Download the data.

The model will probably train faster if the data is local.

You can download the data from these links:

*   [train](https://storage.googleapis.com/unsupervised-hierarch-video/data/humans-train)
*   [validation](https://storage.googleapis.com/unsupervised-hierarch-video/data/humans-validation)
*   [test](https://storage.googleapis.com/unsupervised-hierarch-video/data/humans-test)

Add the `--data_dir` flag to the following commands with the location of the
data.

### Robot push

You'll have to download the robot push dataset
[here](https://storage.googleapis.com/unsupervised-hierarch-video/robot_data.tar.gz)
to train with it.

Add the `--data_dir` flag to the following commands with the location of the
data.

## Commands

These commands have the best known hyperparameters for each mode and dataset.
The learning rate and batch size were optimized for parallel async training on
32 GPUs, so they may not be optimal for training on a single GPU.
Below, we provide the hyperparameters for single GPU training on the Human 3.6M
dataset.

For all commands, set "model_dir" and "event_log_dir" to the location where the
model and tensorboard logs should be saved.

Use the same commands for validation except:

*   Add the nois_training flag and set run_mode to "eval".

*   Set data_pattern to "*validation*"

*   Remove the enc_keep_prob and van_keep_prob flags.

*   Chang the event_log_dir to what you want to use for eval.

### Humans dataset

#### EPVA Gan (Multi-GPU Hyperparameters):

This command starts training from a pretrained EPVA checkpoint. Follow the steps
above to install gcloud and sign in to use the command EPVA gan command.
Alternatively, download the checkpoint [here](https://storage.googleapis.com/unsupervised-hierarch-video/pretrained_models/epva_human.zip) or with `gsutil cp -r
gs://unsupervised-hierarch-video/pretrained_models/epva_human ~/Downloads` and
pass `--epv_pretrain_ckpt <Download path>`.

`python prediction_train.py --model_mode epva_gan --enc_learning_rate 1e-5
--pred_learning_rate_map 1e-06 --van_learning_rate 3e-06 --discrim_learning_rate
3e-06 --enc_pred_loss_scale 10 --enc_size_set 64 --enc_keep_prob .65
--van_keep_prob .9 --batch_size 16 --sequence_length 64 --skip_num 2
--context_frames 5 --run_mode "train" --is_training --train_steps 1000000
--clip_gradient_norm .01 --pred_noise_std 1.0 --enc_pred_use_l2norm`

#### EPVA Gan (Single-GPU Hyperparameters):

Coming soon....

#### EPVA (Multi-GPU Hyperparameters):

This command starts training the encoder from a pretrained imagenet checkpoint.
You have the same option to either run as is or download the checkpoint. You can
download the checkpoint
[here](https://storage.googleapis.com/unsupervised-hierarch-video/pretrained_models/vgg_16.ckpt),
and change the `checkpoint_path` variable in `prediction_train`.

`python prediction_train.py --model_mode epva --imgnet_pretrain
--all_learning_rate 1e-05 --enc_pred_loss_scale .1 --enc_pred_loss_scale_delay
6e5 --enc_size_set 64 --enc_keep_prob .65 --van_keep_prob .9 --batch_size 8
--sequence_length 64 --skip_num 2 --context_frames 5 --run_mode "train"
--is_training --train_steps 3000000 --clip_gradient_norm .01 --epv_pretrain_ckpt
''`

#### EPVA (Single-GPU Hyperparameters):

This command starts training the encoder from a pretrained imagenet checkpoint.
You have the same option to either run as is or download the checkpoint. You can
download the checkpoint
[here](https://storage.googleapis.com/unsupervised-hierarch-video/pretrained_models/vgg_16.ckpt),
and change the `checkpoint_path` variable in `prediction_train`.

`python prediction_train.py --model_mode epva --imgnet_pretrain
--all_learning_rate 1e-04 --enc_pred_loss_scale .1 --enc_pred_loss_scale_delay
2e4 --enc_size_set 64 --enc_keep_prob .65 --van_keep_prob .9 --batch_size 8
--sequence_length 64 --skip_num 2 --context_frames 5 --run_mode "train"
--is_training --train_steps 120000 --clip_gradient_norm .01 --epv_pretrain_ckpt
''`


#### E2E:

`python prediction_train.py --model_mode e2e --dataset_type human
--all_learning_rate 1e-05 --enc_size_set 64 --enc_keep_prob .65 --van_keep_prob
.9 --batch_size 8 --sequence_length 64 --skip_num 2 --context_frames 5
--run_mode "train" --is_training --train_steps 3000000 --clip_gradient_norm .01
--epv_pretrain_ckpt ''`

### Robot Push dataset

E2E:

`python prediction_train.py --model_mode e2e --dataset_type robot
--all_learning_rate 1e-05 --enc_size_set 16 --enc_keep_prob .65 --van_keep_prob
.9 --batch_size 8 --sequence_length 20 --skip_num 1 --run_mode "train"
--is_training --train_steps 3000000 --clip_gradient_norm .01 --epv_pretrain_ckpt ''`

EPVA:

`python prediction_train.py --model_mode epva --dataset_type robot
--imgnet_pretrain --all_learning_rate 1e-05 --enc_pred_loss_scale 1
--enc_pred_loss_scale_delay 6e5 --enc_size_set 32 --enc_keep_prob .65
--van_keep_prob .9 --batch_size 8 --sequence_length 20 --skip_num 1 --run_mode
"train" --is_training --train_steps 3000000 --clip_gradient_norm .01 --epv_pretrain_ckpt ''`

Individual:

`python prediction_train.py --model_mode individual --dataset_type robot
--enc_learning_rate 1e-5 --pred_learning_rate_map 3e-4 --van_learning_rate 3e-5
--enc_size_set 12 --enc_keep_prob .75 --van_keep_prob 1.0 --batch_size 8
--sequence_length 20 --skip_num 1 --run_mode "train" --is_training --train_steps
3000000 --clip_gradient_norm .01 --epv_pretrain_ckpt ''`

## Inference on pretrained models.

[Install Jupyter](http://jupyter.org/install) and open the generation.ipynb
file.

You can use this to run inference on pretrained models saved to Google Cloud, or
modify it to do inference on your own models.

## Saved images.

You can download the images produced by running the inference in the following
links. They are the original 64x64 resolution, and have about 128 frames of
prediction (about 20 seconds.)

*   [Ground
    truth](https://storage.googleapis.com/unsupervised-hierarch-video/generated_images/human_gt.tar.gz)
*   [EPVA
    Gan](https://storage.googleapis.com/unsupervised-hierarch-video/generated_images/human_epva_wgan.tar.gz)
*   [EPVA](https://storage.googleapis.com/unsupervised-hierarch-video/generated_images/human_epva.tar.gz)
*   [E2E](https://storage.googleapis.com/unsupervised-hierarch-video/generated_images/human_e2e.tar.gz)

There are also images for other methods in the paper which can be found with
`gsutil ls gs://unsupervised-hierarch-video/generated_images`

## Notes

Contact wichersn@google.com or file an issue if you have any questions or
comments.

This is not an official Google product.

[^1]: Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu.
    Human3.6m: Large scale datasets and predictive methods for 3d human
    sensing in natural environments. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 36(7):1325–1339, jul 2014.
[^2]: Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for
    physical interaction through video prediction. In Advances in Neural
    Information Processing Systems, pp. 64–72, 2016.
