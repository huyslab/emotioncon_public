# Emotion Regulation Experiment Data Analysis

This repository contains the code for analyzing the data collected in an emotion regulation experiment. The project involves utilizing various Jupyter notebooks, each serving a specific purpose, to gain insights into the dynamics of emotion ratings. Below is an overview of the notebooks included in this project:

## Notebooks Overview

- `kalman_simulations.ipynb`: Conducts simulations to determine the optimal length of the experiment using advanced Kalman filtering techniques.

- `prepdata.ipynb`: Downloads the data from Firebase and preprocesses it for subsequent analyses and modeling.

- `emotion_dynamics.ipynb`: Visualizes the emotion ratings and explores their intricate dynamics, including their overlap with the original ratings from the Cowen & Keltner paper (PNAS, 2017).

- `basics_characteristics.ipynb`: Calculates essential characteristics of the data, such as mean and variance before and after intervention, while exploring their potential relationship with psychiatric symptoms and test-retest of ratings.

- `modelling_experiment.ipynb`: Focuses on fitting a Kalman filter to the data and compares different models to determine whether the different components of the Kalman Filter are necessary to provide a parsimonious account of the emotion ratings.

- `parameter_recovery.ipynb`: Explors the recoverability of the dynamics and input weight matrix in the most parsiminous model.

- `replicate_mean_effect.ipynb`: Replicates the distancing effect in mean ratings by adapting the dynamics or the
input weights alone in simulations.

- `modelling_analysis.ipynb`: Delves into the analysis of the estimates of the dynamics obtained from the Kalman filter, including comparisons between intervention groups, such as dynamics and controllability features and potential relationship with psychiatric symptoms.

9. `paper_plots.ipynb`: Generates additional figures for the paper to provide visual support for the obtained results.


## Prerequisites

To run the code in this project, make sure you have installed Conda, and you can create the necessary environment using the `environment.yml` file provided.

## Data Access

The raw data used for this analysis is stored on `../data/`.




