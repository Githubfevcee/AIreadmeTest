# Human-Motion-AI
Human-Motion-AI aims to simulate human movement behavior using artificial inteligence. The Unity project is based on the [Walker scenario](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#walker) from the [Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (also known as ml-agents) and incorporates various optimizations. 

**Please note that this documentation keeps track of all steps performed, including those that were unsuccessful. This allows inexperienced developers to evaluate and understand the procedure to gain new knowledge. This documentation captures parts of my learning progress in machine learning and specifically about reinforcement learning in Unity and is intended to be helpful for developers working on similar projects. My recommendation is to read the complete documentation first to prevent avoidable errors. Background knowledge, helpful tips and links are included.**

## Steps of Optimisation
- [x] Training
- [x] Configuration of the Neural Network
- [x] Scripts
- [ ] Implementation of [Marathon](https://github.com/Unity-Technologies/marathon-envs)

The project is work in progress. Unprocessed optimisation methods may be subject of change. 

## Training
In order to generate a custom AI model, it is advisable to first optimize the training process and tailor it to the available hardware.

### Hardware Utilization

Following components were used creating Human-Motion-AI:
- CPU: AMD Ryzen 9-5900X[^1]
- GPU: RTX 4090[^2]
- RAM: 32GB[^3]
[^1]: https://www.amd.com/de/products/cpu/amd-ryzen-9-5900x
[^2]: https://manli.com/en/product-detail-Manli_GeForce_RTX%C2%AE_4090_Gallardo_(M3530+N675)-312.html
[^3]: https://www.corsair.com/de/de/Kategorien/Produkte/Arbeitsspeicher/VENGEANCE%C2%AE-LPX-32GB-%282-x-16GB%29-DDR4-DRAM-3000MHz-C16-Memory-Kit---Black/p/CMK32GX4M2D3000C16

> "For most of the models generated with the ML-Agents Toolkit, CPU will be faster than GPU." -[Unity documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Unity-Inference-Engine.md)

> "This PPO implementation is not optimized for the use of a GPU. In general, it is not that easy to optimize Reinforcement Learning for the use of a GPU. So you are better of with a CPU currently." -[Marco Pleines, PhD student at TU Dortmund, Deep Reinforcement Learning](https://github.com/Unity-Technologies/ml-agents/issues/1246)

It appears that a powerful CPU is crucial for both training and inference, and contrary to popular belief, the GPU plays a subordinate role in this case, because the implemention of the [reinforcement learning  algorithm "PPO"](https://github.com/yosider/ml-agents-1/blob/master/docs/Training-PPO.md) is supposedly not optimized for GPU usage.[^4][^5] The described setup is powerful enough to support a high number of concurrent training processes.
[^4]:https://github.com/Unity-Technologies/ml-agents/issues/1246
[^5]:https://github.com/Unity-Technologies/ml-agents/issues/4129

### Customized Training Routines
#### Attempt 1 - Increased Agent Number 
The initial attempt to speed up the training process was to increase the number of agents from 10 to 20. However, this did not yield any performance improvement during testing, and therefore, this method was discarded.

#### Attempt 2 - Concurrent Unity Instances
In the second attempt [Concurrent Unity Instances](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md#training-using-concurrent-unity-instances) was used. 
Four environments were created, each with 10 agents undergoing training. Therefore, 40 agents were trained across 4 environments. Since each training session runs at 20 times the normal speed, a speed factor of 800 was achieved. At 10 seconds of real-time training the AI model was able to train for 133.3 minutes. Using the no-graphics label is useful for saving hardware resources if there is no need to observe the AI training progress graphically. You can still view a detailed live report of the ongoing training with the [TensorFlow utility named tensorboard](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Using-Tensorboard.md). The result of this training are in detail stored at this uploaded [tensorboard](https://tensorboard.dev/experiment/9a0ykmWaRj2aoi56K9X2hA/#scalars). The official definition for the given diagramms can be found [here](https://unity-technologies.github.io/ml-agents/Using-Tensorboard/).
After almost 7 hours of real-time training (equivalent to 5600 hours of AI training), the following main observations were noted:

- High learning progress during the first hour and a half.
- The middle phase was characterized by low learning progress and even a decline in the success progress.
- Towards the end of the training phase, there was a slow but steady improvement in the success performance.

After completing the training and testing the new AI model, the outcome was disappointing. It did not behave more realistically than the sample AI model provided by the machine learning agents toolkit.

Code insights:  
Starting the learning process with concurrent Unity instances:`mlagents-learn config/ppo/Walker.yaml --env=C:\Users\username\Desktop\Walker\foldername\UnityEnvironment --num-envs=4 --run-id=MyOwnIdentifier --no-graphics`  
Observing the learning process live via tensorboard:`tensorboard --logdir=C:\Users\username\Documents\GitHub\ml-agents\results\MyOwnIdentifier\Walker`

#### Attempt 3 - More Simultaneous Instances and Hidden Units Changes
For this training run the number of environments was increased to 6, which is the maximum supported by the given setup. During the developement phase of this project, a [rare update #5911](https://github.com/Unity-Technologies/ml-agents/pull/5911) (Apr 27, 2023) was released in the developer branch of ml-agents which affected the Walker scenario. Upon installing the new, potentially unstable version and reviewing the revised scenario, a significant improvement in the realism of the sample was observed. Unfortunately, there was no way to train this new model as there was no documentation from Unity regarding this pull, and all commonly used library versions were incompatible[^16]. Since there was no solution to make the new Machine Learning Agents Toolkit version run, it was necessary to review all code modifications. After evaluating which code sections should also work in the old version, they were implemented. The hyperparameters and scripts were reviewed and adopted with modifications. 

**Before proceeding, please review the changes made in the respective sections [Configuration of the Neural Network](https://github.com/georghauschild/AIreadmeTest/edit/main/README.md#configuration-of-the-neural-network) and [Scripts](https://github.com/georghauschild/AIreadmeTest#scripts)!**
[^16]:(https://github.com/Unity-Technologies/ml-agents/issues/5912)


## Configuration of the Neural Network
Before an AI model can be trained, it needs to receive information on how the training will be implemented and executed. The [content of the .yaml file](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md) contains all the relevant information. Some of the most important parameters include: algorithm, [hyperparameters](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/#behavior-configurations), reward signals, and training duration.

### Configuration Modification 1 - Reduced Hidden Units

Excerpt from the .yaml file:  
     
    
    trainer_type: ppo  
    hyperparameters:  
      batch_size: 2048  
      buffer_size: 20480  
      learning_rate: 0.0003  
      beta: 0.005  
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 3
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.995
        strength: 1.0

Reducing the number of hidden units from 512 to 256 represents a significant modification to the AI model, which may lead to faster network training and reduced memory usage, as fewer parameters need to be trained. Additionally, the network may become less susceptible to overfitting, especially if the original network architecture was too large.
On the other hand, decreasing the number of hidden units may result in the network not being able to capture complex relationships in the data as effectively, leading to a poorer model.  
How it will ultimately behave will be shown in [Attempt 3 - More Simultaneous Instances and Hidden Units Changes](https://github.com/georghauschild/AIreadmeTest/edit/main/README.md#attempt-3---more-simultaneous-instances-and-hidden-units-changes), which  describes the impact of this modification during the training and the occurred result. Please proceed to read [Attempt 3 - More Simultaneous Instances and Hidden Units Changes](https://github.com/georghauschild/AIreadmeTest/edit/main/README.md#attempt-3---more-simultaneous-instances-and-hidden-units-changes) for a comprehensive understanding of the actual behavior of the model.


## Scripts

## Software and Library Versions
A well-known and time-consuming issue is getting the framework to run, especially for training purposes. The following versions of the libraries work seamlessly together. Please follow the [official installation instructions](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md) and select the versions listed below.
Creating and using a [virtual environment](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Using-Virtual-Environment.md) has proven to be useful. It prevents version conflicts and will potentially save a significant amount of time.
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
