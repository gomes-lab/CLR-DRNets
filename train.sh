python3 visual_cGAN_9by9.py --load 0 --save 1 --save_unsolve 0 --train_st [START_HINTS] --train_ed [END_HINTS] --load_from 1 --batch_size 40 --clip_step 100000 --epoch 1001 --mode solve --drnet 0 --load_from 1 --mix_ratio 0 --no_random 1 --model_number 100 --satnet 0 --update_resnet 1 --load_name fix_1825- --train_st 17 --resnet_gap 4 --lr 0.0001

# Here I will introduce some meaning of the important arguments
# clip_step: We select the first CLIP_STEP model from the load data
# load: whether we load a pretrained model
# save: whether we save the model
# train_st, train_ed: the number of hints range used to train, note in test mode, only train_st is meaningful
# mode: train or solve
# model_number: how many restart model we used in test
# resnet_gap: how many epoches we reset resnet to the initial weights
# load_name: load model name
