# Autonomous Driving in Smart Cities – Deep Learning for Traffic Optimization

This repository contains the code, diagrams, presentation, and research paper for the project **“Autonomous Driving in Smart Cities: Deep Learning for Traffic Optimization”**.

---

## Repository Structure

- **src/**  
  Contains all source code related to data preprocessing, model definition, training, and evaluation.

- **UML_Diagrams/**  
  Contains the UML diagram used in the research paper, that includes the Activity Diagram 

- **presentation/**  
  Contains the final presentation slides used for project defense.

- **paper/**  
  Contains the research paper (PDF).

---

## Code Overview

The project implements a deep learning–based framework for highway traffic prediction and optimization using the HighD dataset.

Key components include:
- Data loading and preprocessing of HighD recordings
- Traffic state aggregation at the lane–segment level
- Multi-task LSTM model for speed prediction and lane-change activity estimation
- Rule-based traffic optimization using variable speed limits
- Closed-loop evaluation and performance analysis

---

## Main Entry Point

The **main implementation and experimentation code** is provided in the **main Jupyter notebook**:

- `main.ipynb`

This notebook orchestrates the entire workflow, including data preparation, model training, evaluation, and visualization of results.

---

##  Notes

- The code is modular and organized for clarity and reproducibility.
- All experiments reported in the paper are generated from the main notebook.

