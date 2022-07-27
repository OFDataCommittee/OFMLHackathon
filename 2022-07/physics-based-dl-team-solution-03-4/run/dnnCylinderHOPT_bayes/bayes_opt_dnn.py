import subprocess
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import argparse
import time
import pandas as pd
import os

parser = argparse.ArgumentParser(description='Bayesian optimization of hyperparameters for a dnnPotentialFoam case.')
parser.add_argument('N_iter', metavar='N_iter', type=int,
                    help='Number of optimization iterations.')
parser.add_argument("-case","--case", help="Foam case, default is cwd", default = os.getcwd())
args = parser.parse_args()

case = args.case
N_iter=args.N_iter
#max_iterations=2000
#space_lists=['layers', 'nodes_per_layer', 'optimizer_step']
#
#pbounds = {space_lists[0]: (4, 7),
#           space_lists[1]: (10, 30),
#           space_lists[2]: (1e-4, 1e-3)}

space_lists=['layers', 'neurons']

pbounds = {space_lists[0]: (3, 10),
           space_lists[1]: (5, 30)}

cases = ["run_orig"] 
max_parallel_processes = 1

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

def make_integer(dict_):
    new_dict=dict_.copy()
    for key in dict_.keys():
        if key=='optimizer_step':
            break

        new_dict[key]=int(dict_[key])

    return (new_dict)

active_processes = []
suggested=[]
i=0
import os, glob
for filename in glob.glob(os.path.join(case,'dnnPotentialFoam*.csv')):
    os.remove(filename) 
    
def black_box_function(layers,neurons):
    global i
    print(f'------- Optimizer iteration: {i} -------')
    layers = int(layers)
    neurons = int(neurons)
    #next_point_to_probe=make_integer(next_point_to_probe)
    # Construct the hiddenLayers option for pinnFoam
    hidden_layers = ("(")
    for layer in range(layers):
        hidden_layers = hidden_layers + str(neurons) + " "
    hidden_layers = hidden_layers + ")"

    # Start the training
    call_list = ['dnnPotentialFoam',
        '-case', case,
        '-hiddenLayers', hidden_layers,
        #'-maxIterations', str(max_iterations)
        ]
        #'-optimizerStep', str(next_point_to_probe['optimizer_step'])]

    call_string = " ".join(call_list)
    print(f'Probing parameters: {layers,neurons}')
    active_processes.append(subprocess.Popen(call_list, stdout=subprocess.DEVNULL))
    time.sleep(0.5)
    file_name='dnnPotentialFoam-{:08d}.csv'.format(i)
    #time.sleep(1)
    #file_name='dnnPotentialFoam-00000000.csv'
    df = pd.read_csv(file_name)
    target_value=df['TRAINING_MSE'].iloc[-1]
    print(f'Score: {target_value}')
    i=i+1
    # If max number of parallel processes is reached
    if (i % max_parallel_processes == 0):
        # Wait for active processes to finish
        for process in active_processes:
            process.wait()
        active_processes.clear()
    return -target_value


optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
    verbose=0
)

optimizer.maximize(
    init_points=0,
    n_iter=N_iter-1,
)

print('\n ------- Optimal values: ------- \n')
print(f"Layers: {int(optimizer.max['params']['layers'])}")
print(f"Neurons: {int(optimizer.max['params']['neurons'])}")

