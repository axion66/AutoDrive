import os
import pathlib
from typing import Callable

# stable_baseline imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from exports import export_model_as_onnx
from godot_rl.core.utils import can_import
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv


# Custom CNN definition
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#https://github.dev/eugenevinitsky/sequential_social_dilemma_games/blob/b01770993af316105ae5aa0ecd0bb95d9ae948c7/run_scripts/sb3_train.py#L169%23L184
#inspired by him!
class CustomCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[64, 2],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        self.conv1 = nn.Conv2d(
            in_channels=3,  
            out_channels=num_frames * 6, 
            kernel_size=3, 
            stride=1, 
            padding=1 
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=num_frames * 6,
            out_channels=num_frames * 12,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat_out = self._get_flattened_size(observation_space)
        self.fc1 = nn.Linear(in_features=self.flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])
        self.factor_net = nn.Linear(14,2)
        
    def _get_flattened_size(self, observation_space: gym.spaces.Box) -> int:
        sample_input = torch.zeros(1, 3, 32, 32) 
        sample_input = F.tanh(self.conv1(sample_input))
        
        sample_input = F.tanh(self.conv2(sample_input))
        sample_input = self.pool(sample_input)
        return int(torch.flatten(sample_input, start_dim=1).shape[1])

    def forward(self, observations) -> torch.Tensor:
        observations = observations['obs']  # dim = (3098,)
        
        factor_info = observations[0:,0:14]  
        camera = observations[0:,14:]  
        
        camera = camera.reshape(-1, 3, 32, 32)  
        features = F.tanh(self.conv1(camera))  
        features = F.tanh(self.conv2(features)) 
        
        features = self.pool(features)

        features = torch.flatten(features, start_dim=1)
        
        features = F.tanh(self.fc1(features))
        features = self.fc2(features) 
        factor_info = self.factor_net(factor_info)
        
        
        return F.tanh(features + factor_info)

# Default values
env_path = None 
experiment_dir = "logs/"
experiment_name = "experiment"
seed = 42
resume_model_path = None#"model_really_really_final2"
save_model_path = "model_really_really_final2"
save_checkpoint_frequency = 1
onnx_export_path = None
timesteps = 1_000_000_000
inference = False
linear_lr_schedule = False
viz = False
speedup = 1

# Functions for handling ONNX export and model saving
def handle_onnx_export():
    if onnx_export_path is not None:
        path_onnx = pathlib.Path(onnx_export_path).with_suffix(".onnx")
        print("Exporting ONNX to:", os.path.abspath(path_onnx))
        export_model_as_onnx(model, str(path_onnx))

def handle_model_save():
    if save_model_path is not None:
        zip_save_path = pathlib.Path(save_model_path).with_suffix(".zip")
        print("Saving model to:", os.path.abspath(zip_save_path))
        model.save(zip_save_path)

def close_env():
    try:
        print("Closing environment")
        env.close()
    except Exception as e:
        print("Exception while closing env:", e)

# Setting up the checkpoint path
path_checkpoint = os.path.join(experiment_dir, experiment_name + "_checkpoints")
abs_path_checkpoint = os.path.abspath(path_checkpoint)

# Prevent overwriting existing checkpoints
if save_checkpoint_frequency is not None and os.path.isdir(path_checkpoint):
    raise RuntimeError(
        abs_path_checkpoint + " folder already exists. "
        "Use a different experiment_dir or experiment_name, or remove the folder."
    )

# Check for inference mode requirements
if inference and resume_model_path is None:
    raise RuntimeError("Using inference requires resume_model_path to be set.")

# Initialize environment
env = StableBaselinesGodotEnv(
    env_path=env_path, show_window=viz, seed=seed, speedup=speedup
)
env = VecMonitor(env)


# Model loading or creating new model
if resume_model_path is None:
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=2, num_frames=6, fcnet_hiddens=[1024, 2]),
        net_arch=[2],
    )
    model = PPO(
        "MultiInputPolicy",
        env=env,
        learning_rate=2.5e-4,
        n_steps=4096, #update period (if n_steps > states of the entire episode, use the given set)
        batch_size=1024,
        n_epochs=30,
        gamma=0.999, # prolly for discount of values? q() = (R(s,a) + sigma[values*gamma])
        gae_lambda= 1.0,
        ent_coef=2e-4,
        max_grad_norm=40,
        target_kl=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log='tf_log_godot_rl/',
        verbose=3,
    )
    model.learn(total_timesteps=5e8)
    
    
else:
    path_zip = pathlib.Path(resume_model_path)
    print("Loading model:", os.path.abspath(path_zip))
    model = PPO.load(path_zip, env=env, tensorboard_log=experiment_dir)

# Inference or training
if inference:
    obs = env.reset()
    for i in range(timesteps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
else:
    learn_arguments = dict(total_timesteps=timesteps, tb_log_name=experiment_name)
    if save_checkpoint_frequency:
        print("Checkpoint saving enabled. Checkpoints will be saved to:", abs_path_checkpoint)
        checkpoint_callback = CheckpointCallback(
            save_freq=(save_checkpoint_frequency // env.num_envs),
            save_path=path_checkpoint,
            name_prefix=experiment_name,
        )
        learn_arguments["callback"] = checkpoint_callback
    try:
        model.learn(**learn_arguments)
    except KeyboardInterrupt:
        print("Training interrupted by user. Will save/export if paths were provided.")

# Final cleanup and saving
close_env()
handle_onnx_export()
handle_model_save()

