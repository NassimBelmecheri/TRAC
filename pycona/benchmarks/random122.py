import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar

def construct_random122():
    """
    Constructs the Random-122 problem instance.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Parameters
    n_vars = 50
    domain_size = 10
    
    # Variables
    grid = cp.intvar(1, domain_size, shape=(1, n_vars), name="grid")

    model = cp.Model()

    # 122 constraints randomly generated (Provided List)
    # Using 'set' syntax in input, converting to tuples for indexing
    constraints_pairs = [
        (11, 23), (4, 35), (15, 23), (18, 29), (22, 32), (16, 32), (22, 43), (24, 37),
        (20, 43), (26, 31), (36, 48), (12, 18), (3, 12), (14, 34), (24, 41), (16, 49),
        (43, 49), (22, 42), (6, 30), (20, 36), (32, 40), (7, 11), (5, 45), (24, 48),
        (26, 49), (3, 21), (11, 39), (7, 41), (3, 49), (23, 43), (39, 42), (28, 39),
        (2, 6), (31, 38), (20, 38), (3, 36), (11, 45), (13, 35), (2, 27), (41, 43),
        (31, 36), (7, 35), (32, 48), (7, 43), (1, 14), (20, 22), (12, 34), (3, 10),
        (30, 49), (18, 47), (15, 18), (10, 21), (7, 49), (28, 41), (2, 35), (31, 40),
        (38, 49), (9, 28), (2, 15), (38, 42), (29, 30), (30, 33), (16, 38), (16, 41),
        (37, 38), (17, 19), (3, 46), (0, 44), (21, 48), (10, 41), (45, 47), (16, 23),
        (8, 18), (42, 44), (2, 20), (14, 19), (2, 23), (8, 19), (19, 33), (8, 11),
        (9, 24), (0, 26), (17, 29), (10, 30), (11, 49), (30, 42), (10, 22), (29, 34),
        (25, 36), (24, 42), (5, 48), (29, 45), (5, 26), (14, 33), (35, 46), (4, 13),
        (18, 45), (8, 25), (3, 40), (26, 44), (34, 39), (16, 34), (5, 20), (2, 37),
        (14, 25), (27, 38), (4, 26), (17, 22), (27, 28), (4, 47), (9, 10), (25, 39),
        (30, 48), (30, 39), (12, 47), (28, 48), (41, 44), (48, 49), (18, 41), (3, 4),
        (7, 45), (29, 38)
    ]

    for i, j in constraints_pairs:
        model += grid[0, i] != grid[0, j]

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language: Simple binary difference
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, language=lang, name="random122")
    oracle = ConstraintOracle(C_T)

    return instance, oracle
