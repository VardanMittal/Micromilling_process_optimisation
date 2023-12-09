from sklearn.tree import export_graphviz
import graphviz
from subprocess import call
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import plot_model

def visualize_model(model):
    plot_model(model, to_file='neural_network.png', show_shapes=True)
    

def rf_visualize(estimator, features_name,target_name):
    export_graphviz(estimator, out_file='tree.dot', feature_names=features_name, class_names=target_name, filled=True, rounded=True)
    with open('tree.dot') as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

def lines_plot(data, x_attr, y_attr, line_attr):
    plt.figure(figsize=(8, 6))

    x = data[x_attr]
    y = data[y_attr]
    lines = data[line_attr]

    for line_value in set(lines):
        x_line = x[lines == line_value]
        y_line = y[lines == line_value]
        plt.plot(x_line, y_line, label=f'{line_attr} {line_value}')

    plt.xlabel(x_attr)
    plt.ylabel(y_attr)
    plt.title(f'Lines based on {line_attr} on {y_attr} vs {x_attr}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_contour(data, x_attr, y_attr, z_attr):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = data[x_attr]
    y = data[y_attr]
    z = data[z_attr]

    ax.plot_trisurf(x, y, z, cmap='Greys', edgecolor='none')

    ax.set_xlabel(x_attr)
    ax.set_ylabel(y_attr)
    ax.set_zlabel(z_attr)

    plt.title(f"3D Contour Plot of {x_attr}, {y_attr}, {z_attr}")
    plt.show()
