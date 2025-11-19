import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def prepare_data():
    rng = np.random.default_rng(seed=123)
    X_train = rng.uniform(-5, 5, size=(2, 500))
    X_train_extend = rng.uniform(-25, 25, size=(1, 500))
    X_train = np.vstack((X_train, X_train_extend))

    X_test  = rng.uniform(-5, 5, size=(3, 500))
    X_test_extend = rng.uniform(-25, 25, size=(1, 500))
    X_test  = np.vstack((X_test, X_test_extend))
    train_bound = X_train[0, :] ** 2 - X_train[1, :] ** 2
    test_bound  = X_test[0, :] ** 2 - X_test[1, :] ** 2
    y_train = X_train[2, :] > train_bound
    y_test  = X_test[2, :] > test_bound

    return X_train, y_train, X_test, y_test

def plot(x, y):
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)

    f = lambda x, y: (X**2 - Y**2)
    Z = f(X, Y)
    
    fig = go.Figure(
        data=[go.Scatter3d(
              x=x[:, 0], 
              y=x[:, 1],
              z=y,
              mode='markers',
              marker=dict(size=1, color='red')      
            ),
              go.Surface(
              x=X,
              y=Y,
              z=Z,
              colorscale='Viridis'
              )]
        ) 
    
    fig.show()

class DeepNeuralNetwork:
    def __init__(self, imput_dim, hidden_dim, output_dim, lr=0.01, seed):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed


    def fit():
        
    def training(x, y):
        rng = np.random.default_rng(self.seed)
        
        w = rng.normal(size=())


if __name__ == "__main__":
    rng = np.random.default_rng(seed=124)

    x = rng.uniform(-5, 5, size=(2, 500))
    y = rng.uniform(-25, 25, size=(1, 500))
    x = np.vstack((x, y))
    print(x.shape)

    distance = x[0, :] ** 2 - x[1, :] ** 2
        
    mask = y >= distance

