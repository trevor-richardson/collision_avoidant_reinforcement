python3.5 testloss.py --use_ca=False  --policy_inp_type=2 --model_dir='/home/trevor/coding/robotic_pain/pain_data/rl_models_results/exp_0/0_models/' --exp_num=1;
wait;
python3.5 testloss.py --use_ca=False  --policy_inp_type=0 --model_dir='/home/trevor/coding/robotic_pain/pain_data/rl_models_results/exp_0/2_models/' --exp_num=2;
wait;
python3.5 testloss.py --use_ca=False  --policy_inp_type=1 --model_dir='/home/trevor/coding/robotic_pain/pain_data/rl_models_results/exp_0/3_models/' --exp_num=3;
wait;
python3.5 testloss.py --use_ca=False  --policy_inp_type=3 --model_dir='/home/trevor/coding/robotic_pain/pain_data/rl_models_results/exp_0/1_models/' --exp_num=4;
wait;
