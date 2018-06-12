START=$(date +%s)
python3.5 ../machine_learning/train.py --training_iterations=200 --update_size=32 --all_hit=False --use_ca=True --policy_inp_type=2 --exp_num=1;
wait;
python3.5 ../machine_learning/train.py --training_iterations=200 --update_size=32 --all_hit=True --use_ca=True --policy_inp_type=2 --exp_num=2;
wait;
python3.5 ../machine_learning/train.py --training_iterations=200 --update_size=32 --all_hit=True --use_ca=False --policy_inp_type=2 --exp_num=3;
wait;
python3.5 ../machine_learning/train.py --training_iterations=200 --update_size=32 --all_hit=True --use_ca=False --policy_inp_type=2 --exp_num=4;
wait;
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
