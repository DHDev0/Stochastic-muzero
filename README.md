# Stochastic MuZero

Pytorch Implementation of [Stochastic MuZero](https://openreview.net/pdf?id=X6D9bAHhBQ1). Base on [Muzero Unplugged](https://github.com/DHDev0/Muzero-unplugged). 

It is suggested to refer to Stochastic MuZero as "unplugged," as setting the reanalyze_ratio to 0 is necessary to achieve Stochastic MuZero. This is because the original "Stochastic MuZero" paper highlights online reinforcement learning, however, as an enhancement to "MuZero Unplugged," it also encompasses offline reinforcement learning capabilities.

[MuZero](https://github.com/DHDev0/Muzero) -> [MuZero Unplugged](https://github.com/DHDev0/Muzero-unplugged) -> [Stochastic MuZero](https://github.com/DHDev0/Stochastic-muzero)

Table of contents
=================
<!--ts-->
   * [Getting started](#getting-started)
      * [Local Installation](#local-installation)
      * [Docker](#docker)
      * [Dependency](#dependency)
   * [Usage](#usage)
      * [Jupyter Notebook](#jupyter-notebook)
      * [CLI](#cli)
   * [Features](#features)
   * [How to make your own custom gym environment?](#how-to-make-your-own-custom-gym-environment)
   * [Authors](#authors)
   * [Subjects](#subjects)
   * [License](#license)

<!--te-->

Getting started
===============

Local Installation
------------------

PIP dependency : [requirement.txt](https://github.com/DHDev0/Stochastic-muzero/blob/main/requirements.txt)
~~~bash
git clone https://github.com/DHDev0/Stochastic-muzero.git

cd Stochastic-muzero

pip install -r requirements.txt
~~~

If you experience some difficulty refer to the first cell [Tutorial](https://github.com/DHDev0/Stochastic-muzero/blob/main/tutorial.ipynb) or use the dockerfile.

Docker
------
 
Build image: (building time: 22 min , memory consumption: 8.75 GB)
~~~bash
docker build -t stochastic_muzero .
~~~ 
(do not forget the ending dot)

Start container:
~~~bash
docker run --cpus 2 --gpus 1 -p 8888:8888 stochastic_muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 stochastic_muzero
#or
docker run --cpus 2 --gpus 1 --memory 2000M -p 8888:8888 --storage-opt size=15g stochastic_muzero
~~~ 

The docker run will start a jupyter lab on https://localhost:8888//lab?token=token (you need the token) with all the necessary dependency for cpu and gpu(Nvidia) compute.

Option meaning:  
--cpus 2 -> Number of allocated (2) cpu core  
--gpus 1 -> Number of allocated (1) gpu  
--storage-opt size=15gb -> Allocated storage capacity 15gb (not working with windows WSL)  
--memory 2000M -> Allocated RAM capacity of 2GB  
-p 8888:8888 -> open port 8888 for jupyter lab (default port of the Dockerfile)  

Stop the container:
~~~bash
docker stop $(docker ps -q --filter ancestor=stochastic_muzero)
~~~ 

Delete the container:
~~~bash
docker rmi -f stochastic_muzero
~~~ 

Dependency
----------
Language : 
* Python 3.8 to 3.10
(bound by the retro compatibility of Ray and Pytorch)

Library : 
* torch 1.13.0
* torchvision 0.14.0
* ray 2.0.1 
* gymnasium 0.27.0
* matplotlib >=3.0
* numpy 1.21.5

More details at: [requirement.txt](https://github.com/DHDev0/Stochastic-muzero/blob/main/requirements.txt)


Usage
=====

Jupyter Notebook
---------------

For practical example, you can use the [Tutorial](https://github.com/DHDev0/Stochastic-muzero/blob/main/tutorial.ipynb).


CLI
-----------

Set your config file (example): https://github.com/DHDev0/Stochastic-muzero/blob/main/config/

First and foremost cd to the project folder:
~~~bash 
cd Stochastic-muzero
~~~

Construct your dataset through experimentation.
~~~bash 
python muzero_cli.py human_buffer config/experiment_450_config.json
~~~

Training :
~~~bash 
python muzero_cli.py train config/experiment_450_config.json
~~~  

Training with report
~~~bash
python muzero_cli.py train report config/experiment_450_config.json
~~~  

Inference (play game with specific model) :
~~~bash 
python muzero_cli.py train play config/experiment_450_config.json
~~~ 

Training and Inference :
~~~bash 
python muzero_cli.py train play config/experiment_450_config.json
~~~  

Benchmark model :
~~~bash
python muzero_cli.py benchmark config/experiment_450_config.json
~~~ 

Training + Report + Inference + Benchmark :
~~~python 
python muzero_cli.py train report play benchmark play config/experiment_450_config.json
~~~  

Features
========

Core Muzero and Muzero Unplugged features:
* [x] Work for any Gymnasium environments/games. (any combination of continous or/and discrete action and observation space)
* [x] MLP network for game state observation. (Multilayer perceptron)
* [x] LSTM network for game state observation. (LSTM)
* [x] Transformer decoder for game state observation. (Transformer)
* [x] Residual network for RGB observation using render. (Resnet-v2 + MLP)
* [x] Residual LSTM network for RGB observation using render. (Resnet-v2 + LSTM)
* [x] MCTS with 0 simulation (use of prior) or any number of simulation.
* [x] Model weights automatically saved at best selfplay average reward.
* [x] Priority or Uniform for sampling in replay buffer.
* [X] Manage illegal move with negative reward.
* [X] Scale the loss using the importance sampling ratio.
* [x] Custom "Loss function" class to apply transformation and loss on label/prediction.
* [X] Load your pretrained model from tag number.
* [x] Single player mode.
* [x] Training / Inference report. (not live, end of training)
* [x] Single/Multi GPU or Single/Multi CPU for inference, training and self-play.
* [x] Support mix precision for training and inference.(torch_type: bfloat16,float16,float32,float64)
* [X] Pytorch gradient scaler for mix precision in training.
* [x] Tutorial with jupyter notebook.
* [x] Pretrained weights for cartpole. (you will find weight, report and config file)
* [x] Commented with link/page to the paper.
* [x] Support : Windows , Linux , MacOS.
* [X] Fix pytorch linear layer initialization. (refer to : https://tinyurl.com/ykrmcnce)
* [X] Support of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 0.27.0  
* [X] The ability to accommodate any number of players with the provision of player cycle information.
* [X] The incorporation of reanalyze buffer(offline learning) and reanalyze ratio functionality.
* [X] The capability to construct human play datasets through experimentation (CLI only).
* [X] The capability to load human play datasets into the Demonstration buffer or Replay buffer for training.
* [X] The ability to specify the amount of sampled action that MCTS should utilize.
* [X] The implementation of a priority scale on neural network and replay buffer priority.
* [X] Various options for bounding, saving, and deleting games from the reanalyze buffer.
* [X] The introduction of the reanalyze_fraction_mode, which allows for the statistical or  
quantitative switch between new game and reanalyze data with a ratio of reanalyze buffer vs replay buffer."
* [X] The implementation of a scaling parameter of the value loss.

Muzero Stochastic new add-on features include: 
* [X] No gradient scaling.
* [X] Add model of afterstate_prediction_function, aftstate_dynamic_function and encoder_function.
* [X] Extend batch with all observation following an initial index.
* [X] Extend mcts with chance node.
* [X] Extend forward pass with afterstate_prediction, aftstate_dynamic and encoder.
* [X] Extend loss function with value_afterstate_loss, distribution_afterstate_loss and vq-vae_commitment_cost.
* [X] [Encoder] The encoder embedding c_e_t is modeled as a categorical variable.
* [X] [Encoder] Selecting the closest code c_t is equivalent to computing the expression one_hot(arg_max(c_e_t)).
* [X] [Encoder] Use of the Gumbel-Softmax reparameterization trick with zero temperature during the forward pass. (meaning you just forward the encoder with random noise during training and without noise during inference. Since the temperature is 0 you don't forward anything)
* [X] [Encoder] A straight-through estimator is used during the backward of the encoder to allow the gradients to flow only to the encoder during the backpropagation.
* [X] [Encoder] There is no explicit decoder in the model and it does not use a reconstruction loss.
* [X] [Encoder] The network is trained end-to-end in a fashion similar to MuZero.

TODO:
* [ ] Hyperparameter search. (pseudo-code available in self_play.py)
* [ ] Training and deploy on cloud cluster using Kubeflow, Airflow or Ray for aws,gcp and azure.



How to make your own custom gym environment?
================================================

Refer to the [Gym documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)

You will be able to call your custom gym environment in muzero after you register it in gym.

Authors  
==========

- [Daniel Derycke](https://github.com/DHDev0)  

Subjects
========

Deep reinforcement learning


License 
=======

[GPL-3.0 license](https://www.gnu.org/licenses/quick-guide-gplv3.html)  



