import random

import cpmpy as cp
import numpy as np
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
import os 
import json

def construct_job_shop_scheduling_problem(n_jobs=10, machines=3, horizon=20, seed=0):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    random.seed(seed)
    max_time = horizon // n_jobs

    
    # Define a unique filename if none provided
    duration = None
    task_to_mach = None
    filename = f"job_shop_j{n_jobs}_m{machines}_h{horizon}_s{seed}.json"

    # 1. Try to LOAD existing data
    if os.path.exists(filename):
        print(f"[*] Loading Job Shop instance from {filename}")
        with open(filename, 'r') as f:
            data = json.load(f)
            duration = data["duration"]
            task_to_mach = data["task_to_mach"]
    else:
        # 2. GENERATE new data if file missing
        print(f"[*] Generating new Job Shop instance and saving to {filename}")
        random.seed(seed)
        max_time = horizon // n_jobs

        duration = [[0] * machines for i in range(0, n_jobs)]
        for i in range(0, n_jobs):
            for j in range(0, machines):
                duration[i][j] = random.randint(1, max_time)

        task_to_mach = [list(range(0, machines)) for i in range(0, n_jobs)]

        for i in range(0, n_jobs):
            random.shuffle(task_to_mach[i])
            
        # Save to JSON for consistency across languages/runs
        with open(filename, 'w') as f:
            json.dump({
                "duration": duration,
                "task_to_mach": task_to_mach
            }, f, indent=2)


    
    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(0, n_jobs)]

    # convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines = set(task_to_mach.flatten().tolist())

    # decision variables
    start = cp.intvar(1, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(1, horizon, shape=task_to_mach.shape, name="end")

    model = cp.Model()

    grid = cp.cpm_array(np.expand_dims(np.concatenate([start.flatten(), end.flatten()]), 0))

    # precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # duration constraints
    model += (start + duration == end)

    # non_overlap constraints per machine
    for m in machines:
        tasks_on_mach = np.where(task_to_mach == m)
        for (j1, t1), (j2, t2) in all_pairs(zip(*tasks_on_mach)):
            m += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    C_T = list(set(toplevel_list(model.constraints)))

    max_duration = np.max((duration))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]] + \
           [AV[0] + i == AV[1] for i in range(1, max_duration + 1)] + \
           [AV[1] + i == AV[0] for i in range(1, max_duration + 1)]

    instance = ProblemInstance(variables=grid, language=lang,
                               name=f"job_shop_jobs{n_jobs}_machines{machines}_horizon{horizon}_seed{seed}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle
