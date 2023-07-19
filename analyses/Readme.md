## Analyses Details

This project involves analyzing data collected in an emotion regulation experiment. The project is composed of several Jupyter notebooks, each with a specific purpose:

- `kalman_simulations.ipynb`: Conducts simulations to infer the optimal length of the experiment using Kalman filtering techniques.
- `loaddata.py`: Downloads raw data from Firebase and saves it to the `../data/` directory.
- `prepdata.ipynb`: Downloads data from Firebase and preprocesses the data for further analyses and modelling.
- `emotion_dynamics.ipynb`: Visualizes the emotion ratings and shows their complex dynamics, including their overlap with original ratings.
- `basics_characteristics.ipynb`: Calculates basic characteristics of the data, such as the mean and variance before and after intervention and their relation to psychiatric symptoms.
- `modelling_experiment.ipynb`: Fits a Kalman filter to the data to model the dynamics of emotion ratings.
- `modelling_analysis.ipynb`: Analyzes the estimates of the dynamics obtained from the Kalman filter e.g. compares dynamics and controllability feature between intervention groups.
- `paper_plots.ipynb`: Generates additional figures for the paper.
