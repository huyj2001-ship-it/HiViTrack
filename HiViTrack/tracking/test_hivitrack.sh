# Different test settings for MixFormer-ViT-b, MixFormer-ViT-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained MixFomrer-online models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- MixViT-B -----------------##########
### LaSOT test and evaluation
#python tracking/test.py hivitr baseline_small --dataset lasot --threads 8 --num_gpus 2 --runid 392  --params__model Hivitr_ep0400.pth.tar
python tracking/test.py hivitr baseline --dataset lasot --threads 30 --num_gpus 3  --runid 391 --params__model Hivitr_ep0391.pth.tar
#python tracking/test.py hivitr baseline_tiny --dataset got10k_test --threads 8 --num_gpus 2 --runid 400
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_tiny_400
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_384_agent_400
#python tracking/test.py hivitr baseline_small --dataset trackingnet --threads 8 --num_gpus 2 --runid 394


#python tracking/test.py hivitr baseline_384 --dataset vasttrack --threads 8 --num_gpus 3 --runid 391
#python tracking/analysis_results.py --dataset_name vasttrack --tracker_param baseline_384_391

# python tracking/test.py hivitr baseline_384 --dataset trackingnet --threads 8 --num_gpus 1  -runid 391 --params__model mixformer_vit_base_online.pth.tar
# python lib/test/utils/transform_trackingnet.py --tracker_name hivitr_online --cfg_name baseline_384_50

### GOT10k test and pack
# python tracking/test.py mixformer_vit_online baseline --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name hivitr --cfg_name baseline_tiny_400

### UAV123
# python tracking/test.py mixformer_vit_online baseline --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online.pth.tar --params__search_area_scale 4.7
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py mixformer_cvt_online baseline --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online_22k.pth.tar --params__search_area_scale 4.45
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline


##########-------------- MixViT-L -----------------##########
### LaSOT test and evaluation
# python tracking/test.py mixformer_vit_online baseline_large --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.55
# python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_small_394

### TrackingNet test and pack
# python tracking/test.py mixformer_vit_online baseline_large --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.6
# python lib/test/utils/transform_trackingnet.py --tracker_name hivitr --cfg_name baseline_small_394

### GOT10k test and pack
# python tracking/test.py mixformer_vit_online baseline_large --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name hivitr --cfg_name baseline_small_400

### UAV123
# python tracking/test.py mixformer_vit_online baseline_large --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.7
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large

### OTB100
#python tracking/test.py mixformer_cvt_online baseline_large --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online_22k.pth.tar --params__search_area_scale 4.6
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large


