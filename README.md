# Getting Started

Main packages' versions
```
'gym==0.16.0' 
'mujoco-py==1.50.0.1' 
'torch==1.2.0+cu92' 
```

# Examples

### Experiment Environments
'Hopper-v2' \
'HalfCheetah-v2' \
'Ant-v2' \
'Walker2d-v2' 

### Training an agent
```
python Main.py -env_name "HalfCheetah-v2" -seed=1 -ada_steps -guided -q_surr
```

### Playing the pretrained agent
```
python Play_Model.py -env_name "HalfCheetah-v2" -seed=1 -render=True
```

