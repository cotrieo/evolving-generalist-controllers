import pandas as pd
import numpy as np
from nn import NeuralNetwork
from ant_v4_own import AntEnv
from walker2d_v4_own import Walker2dEnv
import os


def generate_morphologies(parameter1_range, parameter2_range, step_sizes):
    parameter1_values = np.arange(parameter1_range[0], parameter1_range[1], step_sizes[0])
    parameter2_values = np.arange(parameter2_range[0], parameter2_range[1], step_sizes[1])

    morphologies = np.array(np.meshgrid(parameter1_values, parameter2_values)).T.reshape(-1, 2)

    return morphologies


def gym_render(game, agent, xml_path, parameters, topology, steps):
    s = 0
    total_reward = 0

    if game == AntEnv:
        xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    elif game == Walker2dEnv:
        xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(xml_path, parameters[0], parameters[1])
        env = game(xml_file, render_mode=None, healthy_reward=0)
    else:
        env = game(parameters)

    obs, info = env.reset(seed=s)
    done = False

    x = agent.cpu()
    nn = NeuralNetwork(x.numpy())
    weights = nn.reshape_layers(topology)

    while not done:
        action = nn.feedforward(weights, topology, obs)

        obs, reward, terminated, truncated, info = env.step(action)

        s += 1
        total_reward += reward

        if s > steps:
            break

        done = terminated or truncated

    env.close()

    return -total_reward


def save_dataframe(dataframe, directory, filename):
    dataframe.to_csv(os.path.join(directory, filename), index=False)


def create_directories(path, subdirectories):
    for subdir in subdirectories:
        os.makedirs(os.path.join(path, subdir), exist_ok=True)


def save_dataframes(evals, best, generalist, generalist_evals, info, path):
    subdirectories = ['xbest', 'generalist', 'evals', 'generalist_evals']

    create_directories(path, subdirectories)

    file_names = [
        '{}_evals.csv'.format(info),
        '{}_xbest.csv'.format(info),
        '{}_generalist.csv'.format(info),
        '{}_generalist_evals.csv'.format(info)
    ]

    dataframes = [evals, pd.DataFrame(best), pd.DataFrame(generalist), generalist_evals]

    for dataframe, subdir, filename in zip(dataframes, subdirectories, file_names):
        save_dataframe(dataframe, os.path.join(path, subdir), filename)
