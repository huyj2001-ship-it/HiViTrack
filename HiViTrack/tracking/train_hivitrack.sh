# There are the detailed training settings for MixFormer-ViT-B and MixFormer-ViT-L.
# 1. download pretrained ViT-MAE models (mae_pretrain_vit_base.pth.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer_vit/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-ViT-B
# Stage1: train mixformer without SPM
# --config: 对应模型的配置文件，放在experiments --save_dir:保存结果的路径  --mode single:单卡 multiple:多卡 
# python tracking/train.py --script mixformer_vit --config baseline --save_dir ./result --mode single --nproc_per_node 1
python tracking/train.py --script hivitr --config baseline  --save_dir output/hivitr --mode multiple --nproc_per_node 2
## Stage2: train mixformer_online, i.e., SPM (score prediction module)
# python tracking/train.py --script mixformer_vit_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL
#python tracking/train.py --script hivitr_online --config baseline_384_got --save_dir output/hivitr_online --mode multiple --nproc_per_node 3 --stage1_model output/hivitr/checkpoints/train/hivitr/baseline_384_got/Hivitr_ep0150.pth.tar

### Training MixFormer-L
#python tracking/train.py --script mixformer_vit --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-B_GOT
#python tracking/train.py --script mixformer_vit --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_online --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT_ONLINE --mode multiple --nproc_per_node 8 \
#    --stage1_model /STAGE1/MODEL
