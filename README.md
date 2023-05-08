# Human-Motion-AI
Human-Motion-AI aims to simulate human movement behavior, using artificial intelligence. The Unity project is based on the [walker scenario](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#walker) from the [Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) (also known as ml-agents) and incorporates various optimisations. 

**Please note that this documentation keeps track of all steps performed, including those that were unsuccessful. This allows inexperienced developers to evaluate and understand the procedure to gain new knowledge. This documentation captures parts of my learning progress in machine learning, specifically about reinforcement learning in Unity and is intended to be helpful for developers working on similar projects. My recommendation is to read the complete documentation first to prevent avoidable errors. Background knowledge, helpful tips and links are included. If you want to dive in directly, be sure to pay close attention to the [installation and version instructions](https://github.com/georghauschild/AIreadmeTest/blob/main/README.md#installation-and-version-instructions).**

## Chronological Order
1. [Installation and Version Instructions](https://github.com/georghauschild/AIreadmeTest#installation-and-version-instructions)
2. [Steps of Optimisation](https://github.com/georghauschild/AIreadmeTest#steps-of-optimisation)
3. [Training - Hardware Utilization](https://github.com/georghauschild/AIreadmeTest#hardware-utilization)
4. [Training - Customized Training Routines - Attempt 1 - Increased Agent Number](https://github.com/georghauschild/AIreadmeTest#attempt-1---increased-agent-number)
5. [Training - Customized Training Routines - Attempt 2 - Concurrent Unity Instances](https://github.com/georghauschild/AIreadmeTest#attempt-2---concurrent-unity-instances)
6. [Training - Customized Training Routines - Attempt 3 - More Simultaneous Instances and Hidden Units Changes - Part One](https://github.com/georghauschild/AIreadmeTest#attempt-3---more-simultaneous-instances-and-hidden-units-changes)
7. [Configuration of the Neural Network - Configuration Modification 1 Reduced Hidden Units](https://github.com/georghauschild/AIreadmeTest#configuration-modification-1---reduced-hidden-units)
8. [Training - Customized Training Routines - Attempt 3 - More Simultaneous Instances and Hidden Units Changes - Part Two](https://github.com/georghauschild/AIreadmeTest#part-two)
9. [Developer Branch Integration](https://github.com/georghauschild/AIreadmeTest#developer-branch-integration)
10. ...to be continued

## Steps of Optimisation
- [x] Training
- [x] Configuration of the Neural Network
- [x] Developer Branch Integration
- [ ] Support from non-AI-based tools
- [ ] Implementation of [Marathon](https://github.com/Unity-Technologies/marathon-envs)

The project is work in progress. Unprocessed optimisation methods may be subject of change. 

## Training
In order to generate a custom AI model, it is advisable to first optimize the training process and tailor it to the available hardware.

### Hardware Utilization

Following components were used creating Human-Motion-AI:
- CPU: AMD Ryzen 9-5900X[^1]
- GPU: RTX 4090[^2]
- RAM: 32 GB[^3]
[^1]: https://www.amd.com/de/products/cpu/amd-ryzen-9-5900x
[^2]: https://manli.com/en/product-detail-Manli_GeForce_RTX%C2%AE_4090_Gallardo_(M3530+N675)-312.html
[^3]: https://www.corsair.com/de/de/Kategorien/Produkte/Arbeitsspeicher/VENGEANCE%C2%AE-LPX-32GB-%282-x-16GB%29-DDR4-DRAM-3000MHz-C16-Memory-Kit---Black/p/CMK32GX4M2D3000C16

> "For most of the models generated with the ML-Agents Toolkit, CPU will be faster than GPU." -[Unity documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Unity-Inference-Engine.md)

> "This PPO implementation is not optimized for the use of a GPU. In general, it is not that easy to optimize Reinforcement Learning for the use of a GPU. So you are better of with a CPU currently." -[Marco Pleines, PhD student at TU Dortmund, Deep Reinforcement Learning](https://github.com/Unity-Technologies/ml-agents/issues/1246)

It appears that a powerful CPU is crucial for both training and inference, and contrary to popular belief, the GPU plays a subordinate role in this case, because the implemention of the [reinforcement learning  algorithm "PPO"](https://github.com/yosider/ml-agents-1/blob/master/docs/Training-PPO.md) is supposedly not optimized for GPU usage.[^4][^5]  
To confirm this, the same AI model was trained using both methods. PyTorch in version 1.8.0+cpu (CPU-focused) and 1.8.0+cu111 (GPU-focused) was used. The result was clear in all test runs. The CPU version is capable of performing calculations faster than the GPU Cuda version.  
![pytorch1 8cpuVSgpupng3](https://user-images.githubusercontent.com/37111215/236808865-2cc2bd89-641c-420f-bad5-ef6717e1bd68.png)  
Left side = Training with version 1.8.0+cpu (CPU-focused)  
Right side = Training with version 1.8.0+cu111 (GPU-focused)
[^4]:https://github.com/Unity-Technologies/ml-agents/issues/1246
[^5]:https://github.com/Unity-Technologies/ml-agents/issues/4129

### Customized Training Routines
#### Attempt 1 - Increased Agent Number 
The initial attempt to speed up the training process was to increase the number of agents from 10 to 20. However, this did not yield any performance improvement during testing, and therefore, this method was discarded.

#### Attempt 2 - Concurrent Unity Instances
In the second attempt, [Concurrent Unity Instances](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-ML-Agents.md#training-using-concurrent-unity-instances) were used. 
Four environments were created, each with 10 agents undergoing training. Therefore, 40 agents were trained across 4 environments. Since each training session runs at 20 times the normal speed, a speed factor of 800 was achieved. At 10 seconds of real-time training, the AI model was able to train for 133.3 minutes. Using the no-graphics label is useful for saving hardware resources if there is no need to observe the AI training progress graphically. You can still view a detailed live report of the ongoing training with the [TensorFlow utility named tensorboard](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Using-Tensorboard.md). The result of this training are in detail stored at this uploaded [tensorboard](https://tensorboard.dev/experiment/9a0ykmWaRj2aoi56K9X2hA/#scalars). The official definition for the given diagrams can be found [here](https://unity-technologies.github.io/ml-agents/Using-Tensorboard/).
After almost 7 hours of real-time training (equivalent to 5600 hours of AI training), the following main observations were noted:

- High learning progress during the first hour and a half.
- The middle phase was characterized by low learning progress and even a decline in the success progress.
- Towards the end of the training phase, there was a slow but steady improvement in the success performance.

After completing the training and testing the new AI model, the outcome was disappointing. It did not behave more realistically than the sample AI model provided by the machine learning agents toolkit.

Code insights:  
Starting the learning process with concurrent Unity instances:  
`mlagents-learn config/ppo/Walker.yaml --env=C:\Users\username\Desktop\Walker\foldername\UnityEnvironment --num-envs=4 --run-id=MyOwnIdentifier --no-graphics`  
Observing the learning process live via tensorboard:  
`tensorboard --logdir=C:\Users\username\Documents\GitHub\ml-agents\results\MyOwnIdentifier\Walker`

#### Attempt 3 - More Simultaneous Instances and Hidden Units Changes
##### Part One
For this training run, the number of environments was increased to 6. Further tests have shown that under the given conditions (hardware and software versions), a maximum number of 6 concurrent Unity instances is reasonable. Beyond that, no time gain is achieved. At this time during the development phase of this project, a [rare update #5911](https://github.com/Unity-Technologies/ml-agents/pull/5911) (Apr 27, 2023) was released in the developer branch of ml-agents which affected the Walker scenario. Upon installing the new, potentially unstable version and reviewing the revised scenario, a significant improvement in the realism of the sample was observed. 
Several undocumented version conflicts have delayed work on the new version[^16]. Meanwhile, the new sections of code were evaluated and partially integrated into the old version of ML Agents. After evaluating which new sections should work in the old version, the neural network settings were adopted. 

**Before proceeding, please review the first change made in the respective section [Configuration of the Neural Network](https://github.com/georghauschild/AIreadmeTest#configuration-of-the-neural-network)!**
[^16]:(https://github.com/Unity-Technologies/ml-agents/issues/5912)

##### Part Two
After 75,000,000 steps the training was finished. The time acceleration factor has increased from 800 to 1200 by adding two more instances. Therefore, 10 seconds of training is equivalent to 200 minutes of training for the AI model. The AI model had nearly 1 year (342.5 days) time for training, but just 7h and 6 minutes have elapsed.
`(60 agents * 10 seconds_real_time * 20 factor_time_acceleration) / 60 seconds = 200 virtual_minutes_training in 10 real_seconds`
  
In contrast to the previous attempt, there is now a continuous increase in success rate (Environment/Cumulative Reward). Nonetheless, the Value Loss diagram (Losses/Value Loss) shows the most significant contrast between the two training runs, where it constantly increased in the previous run but consistently decreased in the current run.
> "The mean loss of the value function update. Correlates to how well the model is able to predict the value of each state. This should increase while the agent is learning, and then decrease once the reward stabilizes." -[Unity Documentation](https://unity-technologies.github.io/ml-agents/Using-Tensorboard/)  
> 
The Value Loss diagram indicates how well the model predicts the values of each state evaluation. During learning, the Value Loss diagram should initially increase as the model attempts to improve its prediction capabilities to better understand the environment and make better decisions. However, once the model starts to better understand the environment and make better decisions, the Value Loss should gradually decrease as the model is able to more accurately predict the state evaluations.  
In contrast, a consistently increasing Value Loss diagram indicates that the model is struggling to understand the environment and make better decisions, leading to poorer performance.  

Value Loss - Attempt 2 - Concurrent Unity Instances:
![Losses_Value Loss  v20](https://user-images.githubusercontent.com/37111215/236862558-ada2e212-edb4-462a-a57d-de83f3d8d533.svg)

Value Loss - Attempt 3 - More Simultaneous Instances and Hidden Units Changes:
![Losses_Value Loss 256HU](https://user-images.githubusercontent.com/37111215/236862968-f16f6ae2-f3af-405e-98da-3163d59698f9.svg)

The complete training run 3 can be viewed in this [Tensorboard](https://tensorboard.dev/experiment/BB7YBlNnQkqu51mYxkpFDw/#scalars).  
Since no substantial changes were made other than halving the Hidden Units in the configuration of the neural network, it is now certain that this modification has had a very positive impact on the AI's ability to better understand its environment and make smarter decisions.



## Configuration of the Neural Network
Before an AI model can be trained, it needs to receive information on how the training will be implemented and executed.

### Configuration Modification 1 - Reduced Hidden Units
The [Training Configuration File .yaml](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md) contains all the relevant information.

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
How it will ultimately behave will be shown in Training Attempt 3, which  describes the impact of this modification during the training and the occurred result.  
**Continue to read the second part of [Attempt 3 - More Simultaneous Instances and Hidden Units Changes](https://github.com/georghauschild/AIreadmeTest#part-two) for a comprehensive understanding of the actual behavior of the model.**

## Developer Branch Integration
As the brand-new push in the [developer branch](https://github.com/Unity-Technologies/ml-agents/tree/develop) proved to be extremely functional, the project was shifted to it and left the [release-20](https://github.com/Unity-Technologies/ml-agents/tree/release_20) branch. Be aware of version changes mentioned in the [installation and version instructions](https://github.com/georghauschild/AIreadmeTest/blob/main/README.md#installation-and-versions-instructions). 
 ```
 Version information: 
  ml-agents: 0.30.0,  
  ml-agents-envs: 0.30.0,  
  Communicator API: 1.5.0,  
  PyTorch: 1.8.0+cpu
```
The same training arguments as in the [Training Attempt 3](https://github.com/georghauschild/AIreadmeTest#attempt-3---more-simultaneous-instances-and-hidden-units-changes) are being used as they have shown to provide satisfactory performance:  
`mlagents-learn config/ppo/Walker.yaml --env=C:\Users\username\Desktop\Walker\exe\UnityEnvironment --num-envs=6 --run-id=MyOwnIdentifier --no-graphics`



## Installation and Version Instructions
A well-known and time-consuming issue is getting the framework to run, especially for training purposes. The following versions of the libraries work seamlessly together. Please follow the [official installation instructions](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md) and select the versions listed below.
Creating and using a [virtual environment](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Using-Virtual-Environment.md) has proven to be useful. It prevents version conflicts and will potentially save a significant amount of time.
create a new environment: `python -m venv python-envs\sample-env`  
activate the environment: `python-envs\sample-env\Scripts\activate`

Release 20:
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

Developer Branch up to [#5911](https://github.com/Unity-Technologies/ml-agents/pull/5911):  
- Machine learning agents toolkit developer branch #5911[^19]
- Unity 2021.3.14f1[^7]
- Python 3.9.12[^8]
- Ml-agents python package 0.30[^9]
- pip 23.1.2[^10]
- PyTorch 1.8.0+cpu[^17]
- Ml-agents Unity package 2.3.0-exp.4<sup>display error? it's maybe exp.3</sup>
- Tensorboard 2.13.0[^13]
- Windows 11[^14]
- Numpy 1.21.2[^15]

*It may also work for future iterations of the developer branch, but it has only been tested up to [#5911](https://github.com/Unity-Technologies/ml-agents/pull/5911).*


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
[^17]:https://pytorch.org/get-started/previous-versions/
[^18]:https://docs.unity3d.com/Packages/com.unity.ml-agents@2.3/manual/index.html
[^19]:https://github.com/Unity-Technologies/ml-agents/pull/5911
