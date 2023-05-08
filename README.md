# Human-Motion-AI
Human-Motion-AI aims to simulate human movement behavior using artificial inteligence. The Unity project is based on the [Walker scenario](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md) from the [Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and applies various optimisations. 

**Please note that this documentation tracks all applied steps, regardless of their usefulness. This documentation records the progress of learning and is beneficial for individuals seeking to undertake similar endeavors.**

## Steps of optimisation
- [x] Training environments
- [ ] Hyperparameter
- [ ] Scripts
- [ ] Implementation of [Marathon](https://github.com/Unity-Technologies/marathon-envs)

Unprocessed optimisation methods may be subject of change.

## Training environments
In order to generate a custom AI model, it is advisable to first optimize the training process and tailor it to the available hardware.
- CPU: AMD Ryzen 9-5900X
- GPU: RTX 4090
- RAM: 32GB
> "For most of the models generated with the ML-Agents Toolkit, CPU will be faster than GPU." -[Unity documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Unity-Inference-Engine.md)

> "This PPO implementation is not optimized for the use of a GPU. In general, it is not that easy to optimize Reinforcement Learning for the use of a GPU. So you are better of with a CPU currently." -[Marco Pleines, PhD student at TU Dortmund, Deep Reinforcement Learning](https://github.com/Unity-Technologies/ml-agents/issues/1246)

It appears that a powerful CPU is crucial for both training and inference, and contrary to popular belief, the GPU plays a subordinate role in this case.

The initial attempt to speed up the training process was to increase the number of agents from 10 to 20. However, this did not yield any performance improvement during testing, and therefore, this method was discarded.

In the second attempt [Concurrent Unity Instances](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md#training-using-concurrent-unity-instances) was used. 4 environments were created, each with 10 agents undergoing training. Therefore, 40 agents were trained across 4 environments. Since each training session runs at 20 times the normal speed, a training speed factor of 800 was achieved. At 10 seconds of real-time training the AI model was trained for 133.3 minutes. The result of this training are in detail stored at this [tensorboard](https://tensorboard.dev/experiment/9a0ykmWaRj2aoi56K9X2hA/#scalars).
