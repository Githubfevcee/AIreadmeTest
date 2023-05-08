# Human-Motion-AI
Human-Motion-AI aims to simulate human movement behavior using artificial inteligence. The Unity project is based on the [Walker scenario](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#walker) from the [Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) and incorporates various optimizations. 

**Please note that this documentation keeps track of all steps performed, including those that were unsuccessful. This allows inexperienced developers to evaluate and understand the procedure to gain new knowledge. This documentation captures parts of my learning progress in machine learning and specifically about reinforcement learning in Unity and is intended to be helpful for developers working on similar projects. Background knowledge, helpful tips and links are included.**

## Steps of optimisation
- [x] Training environments
- [ ] Hyperparameter
- [ ] Scripts
- [ ] Implementation of [Marathon](https://github.com/Unity-Technologies/marathon-envs)

The project is work in progess. Unprocessed optimisation methods may be subject of change. 

## Training environments
In order to generate a custom AI model, it is advisable to first optimize the training process and tailor it to the available hardware. Following components were used creating this project:
- CPU: AMD Ryzen 9-5900X[^1]
- GPU: RTX 4090[^2]
- RAM: 32GB[^3]
[^1]: https://www.amd.com/de/products/cpu/amd-ryzen-9-5900x
[^2]: https://manli.com/en/product-detail-Manli_GeForce_RTX%C2%AE_4090_Gallardo_(M3530+N675)-312.html
[^3]: https://www.corsair.com/de/de/Kategorien/Produkte/Arbeitsspeicher/VENGEANCE%C2%AE-LPX-32GB-%282-x-16GB%29-DDR4-DRAM-3000MHz-C16-Memory-Kit---Black/p/CMK32GX4M2D3000C16

> "For most of the models generated with the ML-Agents Toolkit, CPU will be faster than GPU." -[Unity documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Unity-Inference-Engine.md)

> "This PPO implementation is not optimized for the use of a GPU. In general, it is not that easy to optimize Reinforcement Learning for the use of a GPU. So you are better of with a CPU currently." -[Marco Pleines, PhD student at TU Dortmund, Deep Reinforcement Learning](https://github.com/Unity-Technologies/ml-agents/issues/1246)

It appears that a powerful CPU is crucial for both training and inference, and contrary to popular belief, the GPU plays a subordinate role in this case, because the implemention of the [reinforcement learning  algorithm "PPO"](https://github.com/yosider/ml-agents-1/blob/master/docs/Training-PPO.md) is supposedly not optimized for GPU usage.[^4][^5]
[^4]:https://github.com/Unity-Technologies/ml-agents/issues/1246
[^5]:https://github.com/Unity-Technologies/ml-agents/issues/4129

The initial attempt to speed up the training process was to increase the number of agents from 10 to 20. However, this did not yield any performance improvement during testing, and therefore, this method was discarded.

In the second attempt [Concurrent Unity Instances](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md#training-using-concurrent-unity-instances) was used. Four environments were created, each with 10 agents undergoing training. Therefore, 40 agents were trained across 4 environments. Since each training session runs at 20 times the normal speed, a speed factor of 800 was achieved. At 10 seconds of real-time training the AI model was able to train for 133.3 minutes. The result of this training are in detail stored at this [tensorboard](https://tensorboard.dev/experiment/9a0ykmWaRj2aoi56K9X2hA/#scalars). The official definition for the given diagramms can be found [here](https://unity-technologies.github.io/ml-agents/Using-Tensorboard/). 
After almost 7 hours of real-time training (equivalent to 5600 hours of AI training), the following main observations were noted:

- High learning progress during the first hour and a half.
- The middle phase was characterized by low learning progress and even a decline in the learned achievements.
- Towards the end of the training phase, there was a slow but steady improvement in the success performance.

## Software and library versions
- Machine learning agents toolkit release version 20[^6]
- Unity 2021.3.14f1[^7]
- Python 3.9.12[^8]
- Ml-agents python package 0.30[^9]
- pip 23.1.2[^10]
- PyTorch 1.7.1+cu110[^11]
- Ml-agents Unity package 2.0.1[^12]
- Tensorboard 2.13.0[^13]
- Windows 11[^14]
- Numpy 1.21.2[^15]

[^6]:https://github.com/Unity-Technologies/ml-agents/tree/release_20
[^7]:https://unity.com/releases/editor/whats-new/2021.3.14
[^8]:https://www.python.org/downloads/release/python-3912/
[^9]:https://libraries.io/pypi/mlagents
[^10]:https://pypi.org/project/pip/
[^11]:https://pytorch.org/get-started/previous-versions/
[^12]:https://docs.unity.cn/Packages/com.unity.ml-agents@2.0/changelog/CHANGELOG.html#201---2021-10-13
[^13]:https://github.com/tensorflow/tensorboard/releases/tag/2.13.0
[^14]:https://www.microsoft.com/de-de/software-download/windows11
[^15]:https://numpy.org/doc/stable/release/1.21.2-notes.html

Walker Benchmark Mean Reward : 2500

Training with 256 HU:
https://tensorboard.dev/experiment/BB7YBlNnQkqu51mYxkpFDw/#scalars
Update: 6 env not 4 anymore
