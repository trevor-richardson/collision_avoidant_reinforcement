# collision_avoidant_reinforcement
This repository implements a deep reinforcement learning algorithm designed to utilize uncertainty in robotic model dynamics to learn to avoid future collisions. <br/>

This repository implements and explores a new way to train policy gradient algorithms using the output from the self-supervised perturbation detection algorithm presented in my [deep_dynamics](https://github.com/trevor-richardson/deep_dynamics) repository as the reward for each action in a given state. The motivation for this approach is based upon developing new learning algorithms for robots that minimize damaging interactions during machine learning in new environments. This section proposes a new deep reinforcement learning algorithm to attempt to solve the dodge ball task with n projectiles rather than just one. Specifically, the research below was tested on eight projectiles. The Monte-Carlo (i.e. stochastic) policy gradient algorithm method was used, however, the proposed intrinsic reward strategy is not specific to the Monte-Carlo policy gradient algorithm. The intrinsic reward strategy presented is general to any policy gradient method and may be usable in other reinforcement learning frameworks.
<br/>

## Model
<img src="https://github.com/trevor-richardson/collision_avoidant_reinforcement/blob/master/visualizations/deep_intrinsic_rl.png" width="950">

---

### Installing
Change BASE_DIR in config.ini to the absolute path of the current directory. <br/>
Packages needed to run the code include:
* numpy
* scipy
* python3
* pytorch
* matplotlib
* vrep
