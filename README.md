# RDQN: Extending the Deep Q Network to Recurrent Neural Networks

This repository contains Matlab code to train the models explored in my MSc project (pdf file included). Some of the early code was written by [Alexander Botev](https://github.com/Botev).

## Training the models

In order to train the models, first navigate to the root directory in Matlab and add all the nested files using `addpath(genpath('.'))`. Then choose one of the `follow___Policy` files (see below) to train your own neural nets!

## Common files

+ `learnDQN.m` Uses Adam to train the DQN or DQRNN
+ `gameToHistory.m` Converts a cell of games to a cell of transitions (histories), where each transition consists of (s, a, r, s2), as well as time step and an indicator for the terminal state.
+ `evaluateSample.m` A method of both the DQN and DQRNN classes. The two versions are different: the DQN one samples some _transitions_ from a history and computes the error and gradient on them. The DQRNN version samples some entire episodes (where each episode is a game) and computes the error and gradient on the entire episodes. In both cases, the error is computed against a second weight vector for the same architecture: this is the 'target' network. The network that is being updated and the target network are managed by learnDQN.m. The pdf report explains these differences in more detail.

## Specific files

In these files, the ___ can be replaced by Catch, Catch2 or TMan. They are specific to each game.

+ `follow___Policy` follows an epsilon-greedy policy on one of the games. Takes as input a weight vector as well as a DQN.		
+ `run___Experiment` runs an experiment (i.e. learns a DQN/DQRNN). This is the file you will use to train networks. Parameters can be changed to control many features:
..+ `nNewTrans` - whether new experiences are generated (following epsilon-greedy policy) (and how many)
..+ `nFuncEvals` - number of gradient computations to make
..+ `memoryLength` - how long to keep transitions in the memory (measured in number of transitions)

## Useful terminology

A 'session' is played on fixed experience - new transitions are added (if nNewTrans > 0) at the end of each episode. By default, the target network weights are fixed throughout the session and are updated at the beginning of each new session.