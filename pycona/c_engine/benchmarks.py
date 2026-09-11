from typing import Tuple, List, Set, Dict, Any
from .c_interface import (
    CQuAcqEngine,
    OP_NE, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE,
    OP_DIFF_OFFSET_EQ, OP_DIFF_OFFSET_NE,
    OP_ABS_DIFF_EQ, OP_ABS_DIFF_NE,
    OP_GOLOMB_EQ, OP_GOLOMB_NE, OP_GOLOMB_LT, OP_GOLOMB_GT,
    OP_TERNARY_DIFF_EQ,
    OP_UNARY_EQ, OP_UNARY_NE
)

def construct_sudoku(grid_size: int = 4, block_row: int = 2, block_col: int = 2) -> CQuAcqEngine:
    """
    Constructs a Sudoku problem directly on the C++ backbone without CPMpy.
    For standard Sudoku 16: grid_size=4, block_row=2, block_col=2 (16 variables, domain 1..4).
    For standard Sudoku 9x9: grid_size=9, block_row=3, block_col=3 (81 variables, domain 1..9).
    """
    num_vars = grid_size * grid_size
    engine = CQuAcqEngine(num_vars)
    domain_vals = list(range(1, grid_size + 1))

    for i in range(num_vars):
        engine.set_domain(i, domain_vals)

    def var_idx(r, c):
        return r * grid_size + c

    target_pairs: Set[Tuple[int, int]] = set()

    # 1. Row constraints
    for r in range(grid_size):
        for c1 in range(grid_size):
            for c2 in range(c1 + 1, grid_size):
                u, v = var_idx(r, c1), var_idx(r, c2)
                target_pairs.add((min(u, v), max(u, v)))

    # 2. Column constraints
    for c in range(grid_size):
        for r1 in range(grid_size):
            for r2 in range(r1 + 1, grid_size):
                u, v = var_idx(r1, c), var_idx(r2, c)
                target_pairs.add((min(u, v), max(u, v)))

    # 3. Block constraints
    for br in range(0, grid_size, block_row):
        for bc in range(0, grid_size, block_col):
            block_vars = []
            for r in range(br, br + block_row):
                for c in range(bc, bc + block_col):
                    block_vars.append(var_idx(r, c))
            for i in range(len(block_vars)):
                for j in range(i + 1, len(block_vars)):
                    u, v = block_vars[i], block_vars[j]
                    target_pairs.add((min(u, v), max(u, v)))

    # Set target constraints in C++ oracle
    for u, v in target_pairs:
        engine.add_target_constraint(OP_NE, [u, v], 0)

    # Set bias: all pairs with relation language {==, !=, <, >, <=, >=}
    ops = [OP_NE, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE]
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            for op in ops:
                engine.add_bias_constraint(op, [i, j], 0)

    return engine


def construct_nqueens(n: int = 8) -> CQuAcqEngine:
    """
    Constructs an N-Queens problem on the C++ backbone.
    """
    engine = CQuAcqEngine(n)
    domain_vals = list(range(1, n + 1))
    for i in range(n):
        engine.set_domain(i, domain_vals)

    # Target: AllDifferent columns + diagonals
    for i in range(n):
        for j in range(i + 1, n):
            engine.add_target_constraint(OP_NE, [i, j], 0)
            engine.add_target_constraint(OP_DIFF_OFFSET_NE, [i, j], i - j)
            engine.add_target_constraint(OP_DIFF_OFFSET_NE, [i, j], j - i)

    # Bias
    ops = [OP_NE, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE]
    for i in range(n):
        for j in range(i + 1, n):
            for op in ops:
                engine.add_bias_constraint(op, [i, j], 0)
            for offset in range(-n, n + 1):
                engine.add_bias_constraint(OP_DIFF_OFFSET_NE, [i, j], offset)

    return engine


def construct_zebra() -> CQuAcqEngine:
    """
    Constructs the Zebra (Einstein's) puzzle:
    25 variables, domains 1..5, representing houses 1 to 5:
    Categories:
      0: Nationalities (ukr=0, norge=1, eng=2, spain=3, jap=4)
      1: Colors (red=5, blue=6, yellow=7, green=8, ivory=9)
      2: Cigarettes (oldGold=10, parly=11, kools=12, lucky=13, chest=14)
      3: Pets (zebra=15, dog=16, horse=17, fox=18, snails=19)
      4: Drinks (coffee=20, tea=21, h2o=22, milk=23, oj=24)
    """
    engine = CQuAcqEngine(25)
    domain_vals = [1, 2, 3, 4, 5]
    for i in range(25):
        engine.set_domain(i, domain_vals)

    ukr, norge, eng, spain, jap = 0, 1, 2, 3, 4
    red, blue, yellow, green, ivory = 5, 6, 7, 8, 9
    oldGold, parly, kools, lucky, chest = 10, 11, 12, 13, 14
    zebra, dog, horse, fox, snails = 15, 16, 17, 18, 19
    coffee, tea, h2o, milk, oj = 20, 21, 22, 23, 24

    # Target clues
    engine.add_target_constraint(OP_EQ, [eng, red], 0)
    engine.add_target_constraint(OP_EQ, [spain, dog], 0)
    engine.add_target_constraint(OP_EQ, [coffee, green], 0)
    engine.add_target_constraint(OP_EQ, [ukr, tea], 0)
    engine.add_target_constraint(OP_DIFF_OFFSET_EQ, [green, ivory], 1)
    engine.add_target_constraint(OP_EQ, [oldGold, snails], 0)
    engine.add_target_constraint(OP_EQ, [kools, yellow], 0)
    engine.add_target_constraint(OP_UNARY_EQ, [milk], 3)
    engine.add_target_constraint(OP_UNARY_EQ, [norge], 1)
    engine.add_target_constraint(OP_ABS_DIFF_EQ, [chest, fox], 1)
    engine.add_target_constraint(OP_ABS_DIFF_EQ, [kools, horse], 1)
    engine.add_target_constraint(OP_EQ, [lucky, oj], 0)
    engine.add_target_constraint(OP_EQ, [jap, parly], 0)
    engine.add_target_constraint(OP_ABS_DIFF_EQ, [norge, blue], 1)

    # AllDifferent within each category
    for cat in range(5):
        cat_vars = [cat * 5 + i for i in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                engine.add_target_constraint(OP_NE, [cat_vars[i], cat_vars[j]], 0)

    # Bias generation:
    binary_ops = [
        (OP_NE, 0), (OP_EQ, 0), (OP_LT, 0), (OP_GT, 0), (OP_LE, 0), (OP_GE, 0),
        (OP_ABS_DIFF_EQ, 1), (OP_ABS_DIFF_NE, 1),
        (OP_DIFF_OFFSET_EQ, 1), (OP_DIFF_OFFSET_EQ, -1)
    ]
    for i in range(25):
        for j in range(i + 1, 25):
            for op, p in binary_ops:
                engine.add_bias_constraint(op, [i, j], p)

    # Unary candidate constraints
    for i in range(25):
        for c in range(1, 6):
            engine.add_bias_constraint(OP_UNARY_EQ, [i], c)
            engine.add_bias_constraint(OP_UNARY_NE, [i], c)

    return engine


def construct_golomb(n_marks: int = 8) -> CQuAcqEngine:
    """
    Constructs the Golomb Ruler problem for n_marks marks.
    Target constraints:
      - Ordering: X_i < X_j for all i < j
      - All-different distances: |X_j - X_i| != |X_y - X_x| for all distinct pairs (i, j) and (x, y)
    """
    engine = CQuAcqEngine(n_marks)
    max_len = 35 if n_marks <= 8 else n_marks * n_marks
    domain_vals = list(range(1, max_len + 1))

    for i in range(n_marks):
        engine.set_domain(i, domain_vals)

    # 1. Ordering: X_i < X_j for all i < j
    for i in range(n_marks):
        for j in range(i + 1, n_marks):
            engine.add_target_constraint(OP_LT, [i, j], 0)

    # 2. Distinct pairs
    pairs = []
    for i in range(n_marks):
        for j in range(i + 1, n_marks):
            pairs.append((i, j))

    for p1 in range(len(pairs)):
        i, j = pairs[p1]
        for p2 in range(p1 + 1, len(pairs)):
            x, y = pairs[p2]
            engine.add_target_constraint(OP_GOLOMB_NE, [i, j, x, y], 0)

    # Bias generation:
    binary_ops = [OP_NE, OP_EQ, OP_LT, OP_GT, OP_LE, OP_GE]
    for i in range(n_marks):
        for j in range(i + 1, n_marks):
            for op in binary_ops:
                engine.add_bias_constraint(op, [i, j], 0)

    # Ternary relations: X_j - X_i == X_k
    for i in range(n_marks):
        for j in range(i + 1, n_marks):
            for k in range(n_marks):
                if k != i and k != j:
                    engine.add_bias_constraint(OP_TERNARY_DIFF_EQ, [j, i, k], 0)

    # Quaternary distance relations
    for p1 in range(len(pairs)):
        i, j = pairs[p1]
        for p2 in range(p1 + 1, len(pairs)):
            x, y = pairs[p2]
            engine.add_bias_constraint(OP_GOLOMB_NE, [i, j, x, y], 0)
            engine.add_bias_constraint(OP_GOLOMB_EQ, [i, j, x, y], 0)
            engine.add_bias_constraint(OP_GOLOMB_LT, [i, j, x, y], 0)
            engine.add_bias_constraint(OP_GOLOMB_GT, [i, j, x, y], 0)

    return engine


BENCHMARKS = {
    "zebra": lambda **kw: construct_zebra(),
    "sudoku16": lambda **kw: construct_sudoku(4, 2, 2),
    "sudoku4": lambda **kw: construct_sudoku(4, 2, 2),
    "sudoku9": lambda **kw: construct_sudoku(9, 3, 3),
    "nqueens": lambda n=8, **kw: construct_nqueens(n),
    "nqueens4": lambda **kw: construct_nqueens(4),
    "nqueens8": lambda **kw: construct_nqueens(8),
    "golomb": lambda n=8, **kw: construct_golomb(n),
    "golomb4": lambda **kw: construct_golomb(4),
    "golomb8": lambda **kw: construct_golomb(8),
}
