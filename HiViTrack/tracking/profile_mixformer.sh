# MixFormer
python tracking/profile_model.py --script hivitr_online --config baseline_384 --online_skip 200 --display_name 'hivitr_online'
python tracking/profile_model.py --script hivitr --config baseline_384 --online_skip 200 --display_name 'hivitr'
python tracking/profile_model.py --script mixformer_cvt_online --config baseline_large --online_skip 200 --display_name 'Mixformer-CVT-L'
# MixFormer-L
python tracking/profile_model.py --script mixformer_cvt --config baseline_large --online_skip 200 --display_name 'Mixformer-CVT-L'