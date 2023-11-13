# Evolving generalist controllers to handle a wide range of morphological variations

## Overview

This repository contains the implementation for the research paper titled "Evolving generalist controllers to handle a wide range of morphological variations." The paper addresses the limited understanding of robustness and generalisability in neuro-evolutionary methods, specifically focusing on artificial neural networks (ANNs) used in control tasks, such as those applied in robotics.

## Abstract

Neuro-evolutionary methods have proven effective in addressing a wide range of tasks. However, the study of the robustness and generalisability of evolved artificial neural networks (ANNs) has remained limited. This has immense implications in fields like robotics where such controllers are used in control tasks. Unexpected morphological or environmental changes during operation can risk failure if the ANN controllers are unable to handle these changes. This paper proposes an algorithm that aims to enhance the robustness and generalisability of the controllers. This is achieved by introducing morphological variations during the evolutionary process. As a result, it is possible to discover generalist controllers that can handle a wide range of morphological variations sufficiently without the need for information regarding their morphologies or adaptation of their parameters. We perform an extensive experimental analysis on simulation that demonstrates the trade-off between specialist and generalist controllers. The results show that generalists are able to control a range of morphological variations with a cost of underperforming on a specific morphology relative to a specialist. This research contributes to the field by addressing the limited understanding of robustness and generalisability in neuro-evolutionary methods and proposes a method by which to improve these properties.

[![SRzcr.gif](https://s5.gifyu.com/images/SRzcr.gif)](https://gifyu.com/image/SRzcr)
  
[Watch the full demonstration on YouTube](https://www.youtube.com/watch?v=eew4X5gBvLQ&t=13s&ab_channel=WorkingMango)

[Based on Preprint: Triebold, C., Yaman, A. (2023).](https://arxiv.org/abs/2309.10201)

## Running Simulations

To run simulations, use the `exp.py` file. Make sure to follow these steps:

1. Install the necessary dependencies. Run the following command to install the required Python packages:

    ```bash
    pip install evotorch pytorch numpy pandas gym mujoco-py joblib
    ```
2. Adjust experiment parameters (ANN topology, morphology variations, initial bounds, etc.) in the respective JSON files
   - CartPole: cart.json
   - BipedalWalker: biped.json
   - Mujoco Ant: ant.json
   - Mujoco Walker2s: walker.json

2. Run the simulations using the following command:

    ```bash
    python exp.py biped.json
    ```
If you are running Ant or Walker2D experiments the morphology variations need to be unzipped first.


