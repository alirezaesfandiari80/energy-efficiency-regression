# Energy Efficiency Prediction with Deep Learning

A regression project using the [UCI Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency) to predict building **Heating Load** from eight building-design features. The notebook covers exploratory data analysis, a linear baseline, a TensorFlow/Keras multilayer perceptron (MLP), cross-validation, error analysis, and feature importance.

## Project Highlights

- Explored 768 building configurations with 8 input features and 2 target variables.
- Used `Heating Load` as the prediction target.
- Compared a scikit-learn Linear Regression baseline with a ReLU-based MLP trained using Adam and mean squared error.
- Evaluated predictions using MAE, RMSE, and R².
- Used five-fold cross-validation, permutation feature importance, and feature ablation to examine robustness and feature contributions.

## Test-Set Results

Results below are from the recorded 80/20 train/test split (`random_state=42`).

| Model | Features | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 8 | 2.1820 | 3.0254 | 0.9122 |
| MLP | 8 | 0.3923 | 0.5060 | 0.9975 |
| MLP | 4 | 0.3990 | 0.5288 | 0.9973 |
| MLP | 2 | 2.3466 | 3.2910 | 0.8961 |

The eight-feature MLP performed substantially better than the linear baseline on this test split. The four-feature model had similar test performance, whereas the two-feature model performed worse. Small differences between the eight- and four-feature models should not be overinterpreted because they may be affected by training randomness.

### Five-Fold Cross-Validation

| Metric | Mean ± standard deviation |
|---|---:|
| MAE | 0.4432 ± 0.0691 |
| RMSE | 0.6146 ± 0.0763 |
| R² | 0.9962 ± 0.0009 |

## Workflow

1. Load the dataset and inspect its structure and descriptive statistics.
2. Explore target distributions, feature relationships, and correlations.
3. Split the data into training and test sets and standardize features using training data only.
4. Train and evaluate a Linear Regression baseline.
5. Train an MLP with two ReLU hidden layers (64 and 32 units), Adam, MSE loss, and early stopping.
6. Compare actual and predicted values and inspect errors.
7. Evaluate the MLP with five-fold cross-validation.
8. Examine permutation feature importance and compare models using different feature subsets.

## Dataset Features

The input features are Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, and Glazing Area Distribution. The original dataset contains both Heating Load and Cooling Load targets; this project focuses on Heating Load.

## Run the Notebook

1. Clone or download this repository.
2. Create and activate a Python environment.
3. Install dependencies from `requirements.txt`.
4. Open `notebooks/energy_efficiency_regression.ipynb` in Jupyter Notebook or VS Code.
5. Run the notebook cells from top to bottom. The dataset is retrieved using `ucimlrepo`, so an internet connection is needed for the initial download.

## Limitations

- The dataset is relatively small and represents a specific set of simulated building configurations.
- Strong test performance does not establish generalization to substantially different real-world buildings.
- Permutation importance describes model dependence on features in this evaluation setup; it should not be interpreted as causal evidence. Correlated features can also affect individual importance scores.
- Neural-network results can vary with random initialization and training conditions.

## Tools

Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · TensorFlow/Keras · Jupyter Notebook · ucimlrepo
