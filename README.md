# SORL: Automatic Symbolic Option Discovery for Facilitating Deep Reinforcement Learning
* Introduction
* Dependencies
* Project Structure
* Start

## Introduction
Despite of achieving great success in real life, Deep Reinforcement Learning (DRL) is still suffering from three critical issues, which are data efficiency, lack of the interpretability and transferability. Recent research shows that embedding symbolic knowledge into DRL is promising in addressing those challenges. Inspired by this, we introduce a novel deep reinforcement learning framework with symbolic options. This framework features a loop training procedure, which enables guiding the improvement of policy by planning with action models and symbolic options learned from interactive trajectories automatically. The learned symbolic options help doing the dense requirement of expert domain knowledge and provide inherent interpretability of policies. Moreover, the transferability and data efficiency can be further improved by planning with the action models. To validate the effectiveness of this framework, we conduct experiments on two domains, Montezuma's Revenge and Office World respectively, and the results demonstrate the comparable performance, improved data efficiency, interpretability and transferability.

## Dependencies
Clone this repository and create a virtual python environment. Then install dependenvies using pip
```
pip install -r requirements.txt
```
To visualize Montezuma's Revenge on local machine, the library atari_py need to be recomplied with the setting 'USE SDL' being True.

## Project Structure
The whole project is decoupled into the following parts:
* Montezuma : the code for reproduing the result of the Montezuma's Revenge
* Office World : the code for reproduing the result of the Office World Domain

## Start
The parameters of each environment are in argParser.py in each folder.
### Montezuma's Revenge
For training SORL Agent
```
python SORL.py
```
For training HRL Agent
```
python hrl.py
```
For training SDRL Agent, you can refer to 
<https://github.com/daomingAU/MontezumaRevenge_SDRL>
### Office World
For training SORL Agent
```
python SORL.py
```
For training HRL Agent
```
python hrl.py
```
For training SORL Agent in task3 and use the options learned in task1 and task2
```
python SORL_portable.py
```
For training HRL Agent in task3 and use the options learned in task1 and task2
```
python HRL_portable.py
```
