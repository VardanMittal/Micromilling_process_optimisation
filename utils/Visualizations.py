from sklearn.tree import export_graphviz
import graphviz
from subprocess import call

def rf_visualize(estimator, features_name,target_name):
    export_graphviz(estimator, out_file='tree.dot', feature_names=features_name, class_names=target_name, filled=True, rounded=True)
    with open('tree.dot') as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])