import numpy as np
from collections import defaultdict

def extract_blocks(block_tree, block_size):

    matrices = {}
    for block_num in block_tree.keys():
        size = block_size[block_num] + 1
        matrices[block_num] = np.zeros((size, size), np.float64)
        for x,y, val in block_tree[block_num]:
            #print(block_num, x,y,val, block_size[block_num])
            matrices[block_num][x,y] = val
            matrices[block_num][y,x] = val

    return matrices


def read_sdp_solution(filename):
    with open(filename, 'r') as solutionfile:
        lines = [line.rstrip('\n') for line in solutionfile]

    primal_blocks = defaultdict(list)
    primal_block_max = defaultdict(lambda: -1)
    
    dual_blocks = defaultdict(list)
    dual_block_max = defaultdict(lambda: -1)
    
    for i, line in enumerate(lines):
        if i == 0:
            continue
        numbers = line.split(' ')
        prim_dual = int(numbers[0])
        block_num = int(numbers[1])
        x = int(numbers[2]) - 1
        y = int(numbers[3]) - 1
        value = float(numbers[4])

        #print("{} {} {} {} {}".format(prim_dual, block_num, x, y, value))

        if prim_dual == 1:
            dual_blocks[block_num].append((x,y,value))
            dual_block_max[block_num] = max(dual_block_max[block_num], x)
        else:
            primal_blocks[block_num].append((x,y,value))
            primal_block_max[block_num] = max(primal_block_max[block_num], x)

    duals_matrices = extract_blocks(dual_blocks, dual_block_max)
    primal_matrices = extract_blocks(primal_blocks, primal_block_max)

    return duals_matrices, primal_matrices

fan_solution = '/home/fgustafsson/Dropbox/uni/courses/semifinite_opti/examples/fan2_orig.sol'
trivial_lp_solution = '/home/fgustafsson/Dropbox/uni/courses/semifinite_opti/trivial_lp.solution'
mnist_sol = '/home/fgustafsson/Dropbox/uni/courses/semifinite_opti/mnist.sol'
