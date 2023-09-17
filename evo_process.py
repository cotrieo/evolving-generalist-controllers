import joblib
import random
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.algorithms import XNES
import torch
import pandas as pd
import numpy as np
from nn import NeuralNetwork
from utils import gym_render, save_dataframes
from ant_v4_modified import AntEnv
from walker2d_v4_modified import Walker2dEnv
from bipedal_walker_modified import BipedalWalker
from cartpole_modified import CartPoleEnv

class Algo:
    def __init__(self, game, path, xml_path, variations, config, run_id, cluster_id, generation):
        self.game = eval(game)
        self.variations = variations
        self.path = path
        self.xml_path = xml_path
        self.max_eval = generation
        self.cluster_id = cluster_id
        self.run_id = run_id
        self.max_fitness = config["maxFitness"]
        self.steps = config['nStep']
        self.topology = config['NN-struc']
        self.initial_stdev = config['stdev_init']
        self.initial_bounds = config["initial_bounds"]
        self.actors = config['actors']
        self.seed = random.randint(0, 1000000)
        self.parameters = self.variations[0]

    def evaluate(self, agent: torch.Tensor) -> torch.Tensor:

        s = 0
        total_reward = 0

        if self.game == AntEnv:
            xml_file = '{}/Ant_{:.2f}_hip_{:.2f}_ankle.xml'.format(self.xml_path, self.parameters[0],
                                                                   self.parameters[1])
            env = self.game(xml_file, render_mode=None, healthy_reward=0)
        elif self.game == Walker2dEnv:
            xml_file = '{}/Walker_{:.3f}_thigh_{:.3f}_leg.xml'.format(self.xml_path, self.parameters[0],
                                                                      self.parameters[1])
            env = self.game(xml_file, render_mode=None, healthy_reward=0)
        else:
            env = self.game(self.parameters)

        obs, info = env.reset(seed=s)
        done = False

        x = agent.cpu()
        nn = NeuralNetwork(x.numpy())
        weights = nn.reshape_layers(self.topology)

        while not done:

            action = nn.feedforward(weights, self.topology, obs)

            obs, reward, terminated, truncated, info = env.step(action)

            s += 1
            total_reward += reward

            if s > self.steps:
                break

            done = terminated or truncated

        env.close()

        return -total_reward

    def problem(self):

        problem = Problem(
            "min",  # minimize the fitness,
            objective_func=self.evaluate,  # evaluation function
            solution_length=NeuralNetwork.calculate_total_connections(self.topology),
            # NN topology determines the solution_length
            initial_bounds=(self.initial_bounds[0], self.initial_bounds[1]),
            # initial bounds limit the initial solutions
            num_actors=self.actors)  # number of actors that run the evaluation in parallel

        searcher = XNES(problem, stdev_init=self.initial_stdev)

        return searcher

    def comparison(self, agent, mode, i):
        if mode == 1:
            fitness = gym_render(self.game, agent, self.xml_path, self.variations[i], self.topology, self.steps)
            return fitness
        else:
            fitness = gym_render(self.game, agent, self.xml_path, self.variations[i], self.topology, self.steps)
            return fitness

    # main function to run the evolution
    def main(self):

        improved = 0
        generalist_old_dev = 0
        prev_pop_best_fitness = 0
        xbest_weights = 0
        generalist_weights = 0
        generation = 0
        current_pop_best_fitness = self.max_eval
        generalist_average_fitness = self.max_eval
        env_counter = 0
        generalist_fitness_scores = np.zeros(len(self.variations))
        good_fitness_scores = np.zeros(len(self.variations))
        number_environments = []
        bad_environments = []
        generalist_average_fitness_history = []
        fitness_std_deviation_history = []
        generalist_min_fitness_history = []
        generalist_max_fitness_history = []

        searcher = self.problem()
        pandas_logger = PandasLogger(searcher)

        print('Number of Environments: ', len(self.variations))
        logger = StdOutLogger(searcher, interval=50)

        while generation < self.max_eval:

            searcher.step()
            index_best = searcher.population.argbest()
            xbest_weights = searcher.population[index_best].values

            if current_pop_best_fitness > searcher.status.get('best_eval'):
                current_pop_best_fitness = searcher.status.get('best_eval')
                xbest_weights_copy = xbest_weights.detach().clone()
                improved = searcher.status.get('iter')

            if len(self.variations) > 1:
                compare = joblib.Parallel(n_jobs=self.actors)(joblib.delayed(self.comparison)(xbest_weights, 1, i)
                                                              for i in range(len(generalist_fitness_scores)))

                generalist_fitness_scores = np.array(compare)

                new_generalist_average_fitness = np.mean(generalist_fitness_scores)
                generalist_average_fitness_history.append(new_generalist_average_fitness)
                generalist_new_dev = np.std(generalist_fitness_scores)

                fitness_std_deviation_history.append(generalist_new_dev)
                generalist_min_fitness_history.append(np.min(generalist_fitness_scores))
                generalist_max_fitness_history.append(np.max(generalist_fitness_scores))

                if new_generalist_average_fitness <= generalist_average_fitness:
                    generalist_average_fitness = new_generalist_average_fitness
                    generalist_old_dev = generalist_new_dev

                    good_fitness_scores = generalist_fitness_scores.copy()
                    generalist_weights = xbest_weights.detach().clone()

                if (searcher.status.get('iter') - improved) % int(np.ceil(self.max_eval * 0.06)) == 0:

                    if current_pop_best_fitness != prev_pop_best_fitness:
                        prev_pop_best_fitness = current_pop_best_fitness
                    else:
                        good_envs = []

                        for i in range(len(self.variations)):
                            if good_fitness_scores[i] < (generalist_average_fitness + generalist_old_dev):
                                good_envs.append(self.variations[i])
                            else:
                                bad_environments.append(self.variations[i])

                        if len(good_envs) == 0:
                            break
                        elif len(good_envs) == len(self.variations):
                            break

                        self.variations = np.array(good_envs)

                        compare = joblib.Parallel(n_jobs=self.actors)(
                            joblib.delayed(self.comparison)(generalist_weights, 0, i)
                            for i in range(len(self.variations)))

                        generalist_fitness_scores = np.array(compare)
                        new_generalist_average_fitness = np.mean(generalist_fitness_scores)
                        if new_generalist_average_fitness < generalist_average_fitness:
                            good_fitness_scores = generalist_fitness_scores.copy()
                        env_counter = len(self.variations) - 1
                        improved = searcher.status.get('iter')

                        print(' no_envs : ', len(self.variations))

                env_counter += 1
                if env_counter >= len(self.variations):
                    env_counter = 0


            elif len(self.variations) == 1:
                generalist_average_fitness = current_pop_best_fitness
                xbest_weights = xbest_weights_copy
                generalist_weights = xbest_weights

            number_environments.append(len(self.variations))
            generation = searcher.status.get('iter')

            if generalist_average_fitness < self.max_fitness:
                break

        evals = pandas_logger.to_dataframe()

        if len(number_environments) != len(evals):
            number_environments.append(len(self.variations))

        evals['no_envs'] = number_environments

        generalist_evals = pd.DataFrame(
            {'Mean': generalist_average_fitness_history, 'STD': fitness_std_deviation_history,
             'Best': generalist_min_fitness_history, 'Worst': generalist_max_fitness_history})

        info = '{}_{}_{}'.format(self.run_id, self.cluster_id, self.seed)

        save_dataframes(evals, xbest_weights, generalist_weights, generalist_evals, info, self.path)

        return generation, np.array(bad_environments)
