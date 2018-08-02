# collision_avoidant_reinforcement
This repository implements a deep reinforcement learning algorithm designed to utilize uncertainty in robotic model dynamics to learn to avoid future collisions. <br/>

This repository implements and explores a new way to train policy gradient algorithms using the output from the self-supervised perturbation detection algorithm presented in my [deep_dynamics](https://github.com/trevor-richardson/deep_dynamics) repository as the reward for each action in a given state. A pre-trained [collision anticipation](https://github.com/trevor-richardson/collision_anticipation) model can be used when training this intrinsic-RL method. The motivation for this approach is to develop new learning algorithms for robots that minimize damaging interactions during machine learning in new environments. This research proposes a new deep reinforcement learning algorithm that attempts to solve the dodge ball task with eight projectiles. The Monte-Carlo (i.e. stochastic) policy gradient algorithm method was used, however, the proposed intrinsic reward strategy is not specific to the Monte-Carlo policy gradient algorithm. The intrinsic reward strategy presented is general to any policy gradient method and may be usable in other reinforcement learning frameworks. A visual and mathematical depiction of the intrinsic-RL method is shown below.
<br/>

## Model
<img src="https://github.com/trevor-richardson/collision_avoidant_reinforcement/blob/master/visualizations/deep_intrinsic_rl.png" width="950">

---
## Scripts to run
Train intrinsic-RL policy gradient method using collision anticipation and a stateful ConvIRNN network structure.
```
python train.py --use_ca=True --policy_inp_type=3
```
<br/>
<br/>

Demo intrinsic-RL policy gradient method with no collision anticipation and a ConvLSTM network structure.

```
python demo_model.py --use_ca=False --policy_inp_type=1
```

### Installing
Change BASE_DIR in config.ini to the absolute path of the current directory. <br/>
Packages needed to run the code include:
* numpy
* scipy
* python3
* PyTorch
* matplotlib
* VREP
