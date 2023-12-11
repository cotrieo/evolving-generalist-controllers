import joblib
import random
from evotorch import Problem, SolutionBatch
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

class Eval(Problem):
    def __init__(self, game, variations, topology, xml_path, steps, initial_bounds, counter):
        super().__init__(
            objective_sense="max",
            solution_length=NeuralNetwork.calculate_total_connections(topology),
            initial_bounds=(initial_bounds[0], initial_bounds[1]),
        )

        self.variations = variations
        self.env_counter = counter
        self.parameters = self.variations[self.env_counter]
        self.topology = topology
        self.game = game
        self.xml_path = xml_path
        self.steps = steps

    def evals(self, agent: torch.Tensor) -> float:

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

        return total_reward

    def _evaluate_batch(self, solutions: SolutionBatch):
        solutions.set_evals(
            torch.FloatTensor(joblib.Parallel(n_jobs=6)(joblib.delayed(self.evals)(i) for i in solutions.values)))
        if len(self.variations) > 1:
            self.env_counter += 1
        else:
            self.env_counter = 0

        if self.env_counter >= len(self.variations):
            self.env_counter = 0

        self.parameters = self.variations[self.env_counter]

    def comparison(self, agent, i):
        fitness = gym_render(self.game, agent, self.xml_path, self.variations[i], self.topology, self.steps)
        return fitness

    def split(self, good_fitness_scores, generalist_avg_fit, generalist_dev, generalist_weights):
        break_stat = False
        good_envs = []
        bad_envs = []

        for i in range(len(self.variations)):
            if good_fitness_scores[i] < (generalist_avg_fit + generalist_dev):
                good_envs.append(self.variations[i])
            else:
                # add underperformed variations to bin
                bad_envs.append(list(self.variations[i]))

        if len(good_envs) == 0:
            print('No more envs')
            break_stat = True
        elif len(good_envs) == len(self.variations):
            print('No more bad envs')
            break_stat = True

        # replace set of variations with only the good variations and re-check their fitness
        self.variations = np.array(good_envs)

        compare_after = joblib.Parallel(n_jobs=4)(joblib.delayed(self.comparison)
                                                  (generalist_weights, i)
                                                  for i in range(len(self.variations)))

        generalist_scores = np.array(compare_after)
        new_avg_fit = np.mean(generalist_scores)

        self.env_counter = 0

        return break_stat, bad_envs, self.variations, generalist_scores, new_avg_fit


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

    def comparison(self, agent, i):
        fitness = gym_render(self.game, agent, self.xml_path, self.variations[i], self.topology, self.steps)
        return fitness

    # main function to run the evolution
    def main(self):

        problem = Eval(self.game, self.variations, self.topology, self.xml_path, self.steps, self.initial_bounds, 0)
        searcher = XNES(problem, stdev_init=self.initial_stdev)

        improved = 0
        generalist_std = 0
        prev_pop_best_fitness = 0
        generalist_weights = 0
        generation = 0
        current_pop_best_fitness = -self.max_eval
        generalist_avg_fit = -self.max_eval
        generalist_scores = np.zeros(len(self.variations))
        good_fitness_scores = np.zeros(len(self.variations))
        number_environments = []
        bad_environments = []
        generalist_avg_history = []
        general_std_history = []
        general_min_fitness_history = []
        general_max_fitness_history = []

        pandas_logger = PandasLogger(searcher)
        print('Number of Environments: ', len(self.variations))
        logger = StdOutLogger(searcher, interval=1)
        torch.set_printoptions(precision=30)

        while generation < self.max_eval:

            # take one step of the evolution and identify the best individual of a generation
            searcher.step()
            index_best = searcher.population.argbest()
            xbest_weights = searcher.population[index_best].values

            # if current best fitness is smaller than new best fitness replace the current fitness and xbest
            if current_pop_best_fitness < searcher.status.get('best_eval'):
                current_pop_best_fitness = searcher.status.get('best_eval')
                improved = searcher.status.get('iter')
                xbest = xbest_weights.detach().clone()

            # if we are running more than 1 variation
            if len(self.variations) > 1:

                # test xbest on all individuals in the morphology set
                compare = joblib.Parallel(n_jobs=self.actors)(joblib.delayed(self.comparison)(xbest_weights, i)
                                                              for i in range(len(generalist_scores)))

                generalist_scores = np.array(compare)

                # check the average fitness score of the morphologies
                new_avg_fit = np.mean(generalist_scores)

                # log the info about the evolution of the generalist
                generalist_avg_history.append(new_avg_fit)
                generalist_new_std = np.std(generalist_scores)
                general_std_history.append(generalist_new_std)
                general_min_fitness_history.append(np.min(generalist_scores))
                general_max_fitness_history.append(np.max(generalist_scores))

                # if current generalist has a smaller avg score than new generalist replace avg score and weights
                if generalist_avg_fit < new_avg_fit:
                    generalist_avg_fit = new_avg_fit
                    generalist_std = generalist_new_std

                    print('Generalist score: ', generalist_avg_fit)

                    good_fitness_scores = generalist_scores.copy()
                    generalist_weights = xbest_weights.detach().clone()

                # check if evolution has stagnated
                if (searcher.status.get('iter') - improved) % int(np.ceil(self.max_eval * 0.06)) == 0:

                    if current_pop_best_fitness != prev_pop_best_fitness:
                        prev_pop_best_fitness = current_pop_best_fitness
                    else:
                        # if the evolution has stagnated check the generalist fitness scores
                        break_stat, bad_envs, self.variations, generalist_scores, new_avg_fit = problem.split(
                            good_fitness_scores,
                            generalist_avg_fit,
                            generalist_std,
                            generalist_weights)

                        if new_avg_fit < generalist_avg_fit:
                            good_fitness_scores = generalist_scores.copy()

                        if len(bad_envs) > 0:
                            for env in bad_envs:
                                bad_environments.append(env)
                            print(bad_environments)

                        if break_stat == True:
                            break

                        improved = searcher.status.get('iter')
                        print(' no_envs : ', len(self.variations))

            # if there is only one morphology generalist = xbest
            elif len(self.variations) == 1:
                generalist_avg_fit = current_pop_best_fitness
                generalist_weights = xbest

            # track the number of envs
            number_environments.append(len(self.variations))
            generation = searcher.status.get('iter')

            # if desired fitness is found terminate evolution
            if generalist_avg_fit > self.max_fitness:
                print('Found best')
                break

        # data logging
        evals = pandas_logger.to_dataframe()

        if len(number_environments) != len(evals):
            number_environments.append(len(self.variations))

        evals['no_envs'] = number_environments

        generalist_evals = pd.DataFrame(
            {'Mean': generalist_avg_history, 'STD': general_std_history,
             'Best': general_min_fitness_history, 'Worst': general_max_fitness_history})

        info = '{}_{}_{}'.format(self.run_id, self.cluster_id, self.seed)

        save_dataframes(evals, xbest, generalist_weights, generalist_evals, info, self.path)

        return generation, np.array(bad_environments)
