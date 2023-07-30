# Multi-agent Reinforcement Learning for Cooperative Adaptive Cruising Control (MACACC)

This repo implements the state-of-the-art MARL algorithms for Cooperative Adaptive Cruising Control, with observability
and communication of each agent limited to its neighborhood. For fair comparison, all algorithms are applied to A2C
agents, classified into two groups: IA2C contains non-communicative policies which utilize neighborhood information
only, whereas MA2C contains communicative policies with certain communication protocols.

Available IA2C algorithms:

* PolicyInferring: [Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in Neural Information Processing Systems, 2017.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
* FingerPrint: [Foerster, Jakob, et al. "Stabilising experience replay for deep multi-agent reinforcement learning." arXiv preprint arXiv:1702.08887, 2017.](https://arxiv.org/pdf/1702.08887.pdf)
* ConsensusUpdate: [Zhang, Kaiqing, et al. "Fully decentralized multi-agent reinforcement learning with networked agents." arXiv preprint arXiv:1802.08757, 2018.](https://arxiv.org/pdf/1802.08757.pdf)
* MACACC: [Chen, Dong, et al. "Communication-Efficient Decentralized Multi-Agent Reinforcement Learning for Cooperative Adaptive Cruise Control"](tbd)

Available MA2C algorithms:

* DIAL: [Foerster, Jakob, et al. "Learning to communicate with deep multi-agent reinforcement learning." Advances in Neural Information Processing Systems. 2016.](http://papers.nips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning.pdf)
* CommNet: [Sukhbaatar, Sainbayar, et al. "Learning multiagent communication with backpropagation." Advances in Neural Information Processing Systems, 2016.](https://arxiv.org/pdf/1605.07736.pdf)
* NeurComm: Inspired
  from [Gilmer, Justin, et al. "Neural message passing for quantum chemistry." arXiv preprint arXiv:1704.01212, 2017.](https://arxiv.org/pdf/1704.01212.pdf)

Available CACC scenarios:

* CACC Catch-up: Cooperative adaptive cruise control for catching up the leadinig vehicle.
* CACC Slow-down: Cooperative adaptive cruise control for following the leading vehicle to slow down.

## Requirements & Libs

* use conda to establish a conda environment: `conda create --name py38 python=3.8 -y`, then activate the environment
* install basic requirements: `pip install -r requirements.txt`

## Usages

First define all hyperparameters (including algorithm and DNN structure) in a config file
under `[config_dir]` ([examples](./config)), and create the base directory of each experiement `[base_dir]`. Other
comments:

- For each scenario (e.g., Catchup and Slowdown), there are corresponding config files.
- To change the number of CAVs in the platoon, you can change `n_vehicle` in the config file


1. To train a new agent, run

~~~
python main.py --base-dir [base_dir] train --config-dir [config_dir]
~~~

Training config/data and the trained model will be output to `[base_dir]/data` and `[base_dir]/model`, respectively.

2. To access tensorboard during training, run

~~~
tensorboard --logdir=[base_dir]/log
~~~

3. To evaluate a trained agent, run

~~~
python main.py --base-dir [base_dir] evaluate
~~~

Evaluation data will be output to `[base_dir]/eva_data`. Make sure evaluation seeds are different from those used in
training.

4. MACACC (i.e., `ia2c_qconsenet`) is defined in `agents/models/QConseNet`. There are several arguments in `models.py` need to be changed
- `self.r` (Lines 163-164) represents the l_{inf} norm of x, thus we need to change to according to different scenarios. For example, if you want to run Slowdown, then uncomment Line 163 instead.
- There are several update strategies in `models.py` (Lines 263-330), `Original` represent the ConseNet update by taking averages; `MACACC` represents non-quantization; `QMACACC (n) represent the quantized version of MACACC`
- The resolution of quantization is controlled by `bi = self._quantization(wt, k, n=1)` (Line 314)

## Acknowledgement

This codes highly depends on Dr. Chu's codes, please give credits to him at: [deeprl_network
](https://github.com/cts198859/deeprl_network)


