# Predicting Building Energy Efficiency

In this project, I explored how machine learning can predict a building’s heating load based on its design characteristics. I started with a Linear Regression model as a baseline, then built a neural network using TensorFlow/Keras to see how much a nonlinear model could improve the predictions.

The project uses the UCI Energy Efficiency dataset, which contains 768 building configurations. Beyond comparing the two models, I also experimented with different feature subsets to understand how the choice of input features affects prediction accuracy.

## Results

The neural network performed considerably better than the linear baseline on the test set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 2.1820 | 3.0254 | 0.9122 |
| Neural Network (8 features) | 0.3923 | 0.5060 | 0.9975 |
| Neural Network (4 features) | 0.3990 | 0.5288 | 0.9973 |
| Neural Network (2 features) | 2.3466 | 3.2910 | 0.8961 |

The full neural network achieved an R² of **0.9975**, with a mean absolute error of **0.3923**. Interestingly, using four features produced nearly the same performance, while reducing the input to two features led to a noticeable drop in accuracy.

I also used five-fold cross-validation to check whether the model's performance remained consistent across different data splits. The average R² was **0.9962 ± 0.0009**.

## What I Worked On

- Explored the dataset and examined relationships between building features and heating load.
- Built a Linear Regression baseline and a multilayer perceptron with ReLU activation.
- Evaluated the models using MAE, RMSE, and R².
- Used five-fold cross-validation to assess model consistency.
- Applied permutation feature importance and feature ablation to investigate the contribution of input features.

## Tools

Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, Jupyter Notebook.

## Running the Project

Install the required packages using `requirements.txt`, then open `notebooks/energy_efficiency_regression.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

The dataset is downloaded from the UCI repository through `ucimlrepo`, so an internet connection is required for the initial download.
