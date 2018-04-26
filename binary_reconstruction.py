# Homework  2 Semidefinte Optimization
# Fredrik Gustafsson
# Eveline de Swart


import numpy as np


import subprocess
import sys
from collections import defaultdict

def set_of_adjacent_nodes(r, image):
    adjacent_nodes = []
    height, width = image.shape
    
    def get_r_neighbours(x,y):
        neighbours = []
        
        for x1 in range(-r, r+1):
            x_delta = x1 + x
            if x_delta < 0 or x_delta >= height:
                continue
            for y1 in range(-r, r+1):
                y_delta = y1 + y
                if y_delta < 0 or y_delta >= width:
                    continue
                
                if (x_delta, y_delta) == (x, y):
                    continue
                neighbours.append((x_delta, y_delta))
        return neighbours

    for i in range(height):
        for j in range(width):
            for neighbour in get_r_neighbours(i, j):
                # i ~ j is also j ~ i so we can instert them lexiographicaly sorted.
                if (i, j) < neighbour:
                    adjacent_nodes.append(((i, j), neighbour))
                else:
                    adjacent_nodes.append((neighbour, (i, j)))
    
    # Make unique
    return set(adjacent_nodes)


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def convert_coordinates_to_sdpa(nodes, image):
    """
    We have an image with shape (width, height).
    From the list of r-adjcajent nodes ( (x_i, y_i),  (x_j, y_j) )
    
    We need to flatten the image to a width*heigth length vector.
    Further more we need a index starting from 1 going to N = width*height (inclusive).
    
    This functions maps the adjenct list to the flattened, 1 indexed vector format.
    
    """
    
    height, width = image.shape
    
    def to_flat(x, y):
        return width*x + (y + 1)
    
    return [(to_flat(adjacent[0][0], adjacent[0][1]), 
             to_flat(adjacent[1][0], adjacent[1][1])) for adjacent in nodes]


def sdpa_r_neighbours(r, image):
    nodes = set_of_adjacent_nodes(r, image)
    return convert_coordinates_to_sdpa(nodes, image)



def build_sdpa_problem(image, lambga_reg, r_neighbours, filename):
    # Image is assumed to have values in [0, 1]

    # Shift to [-1, 1]
    image = 2.0*image - 1.0    
    x, y = image.shape
    # Flatten the image to a long vector.   
    flat_image = np.ndarray.flatten(image)
    
    # The number of pixels in an image
    N = x*y
    
    # Number of unqiue unit-vector
    X_dim = N + 1
    
    # X_ii = 1 gives N+1 constraints
    
    with open(filename, 'w') as sdpa:
    
        # Number of constrains
        sdpa.write("{}\n".format(X_dim))
    
        # Number of blocks
        sdpa.write("1\n")
    
        # Block sizes
        sdpa.write("{}\n".format(X_dim))

        # RHS of constrains, all ones.
        rhs = ' 1'*X_dim + '\n'
        sdpa.write(rhs)


        # LHS contraints X_ii 
        for i in range(1, X_dim + 1):
            sdpa.write("{} 1 {} {} 1.0\n".format(i,i,i))

        # Cost function
        # X_1,2 = flat_image[0]
        # ...
        # X_1,(N+1) = flat_image[N-1]
        for i, g in enumerate(flat_image):
            sdpa.write("0 1 1 {} {}\n".format(i+2, g))
            
        for i,j in r_neighbours:
            if lambga_reg != 0.0:
                sdpa.write("0 1 {} {} {}\n".format(i+1, j+1, lambga_reg))


def solution_to_image(solution, x, y):  
    # Solution is x*y + 1 long vector.    
    # Skipt first element
    return solution[1:].reshape(x,y)


def find_best_rounded_solution(image, X_chol, samples, lambda_reg, r_neighbours):
    flat_image = np.ndarray.flatten(image)
    ndim = X_chol.shape[0]
    random_unit_vectors = sample_spherical(samples, ndim)
    
    x0 = X_chol[0,:]
    
    best_cost = -10**12
    best_rounded_solution = None
    
    for z in random_unit_vectors.T:
        # Make sure x0 is rounded to 1.
        if np.dot(x0, z) < 0.0:
            z = -z
        x_rounded = np.sign(np.matmul(X_chol, z))

        current_cost = eval_rounded_solution(flat_image, x_rounded, lambda_reg, r_neighbours)
        
        if current_cost > best_cost:
            best_cost = current_cost
            best_rounded_solution = x_rounded

    return best_rounded_solution


def eval_rounded_solution(flat_image, rounded_solution, lambda_reg, r_neighbours): 
    data_fidelity = 2.0 * np.dot(flat_image, rounded_solution[1:])
    
    regularization_terms = 0.0
    
    if lambda_reg == 0.0:
        return data_fidelity
    
    for i, j in r_neighbours:
        data_fidelity += rounded_solution[i]*rounded_solution[j]
    
    return data_fidelity + lambda_reg * regularization_terms


def execute(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)

    # Poll process for new output until finished
    while True:
        nextline = process.stdout.readline().decode()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    # Block here until done.
    output = process.communicate()[0]

    return process.returncode


def reconstruct_image(image, lambda_reg, r, problem_file, solution_file, rounding_iterations):
    
    adj_nodes = set_of_adjacent_nodes(r, image)
    r_neighbours = convert_coordinates_to_sdpa(adj_nodes, image)
    r_neighbours.sort() 
    

    # Build problem sdpa
    build_sdpa_problem(image, lambga_reg=lambda_reg, 
                       r_neighbours=r_neighbours, filename=problem_file)


    # Solve it
    execute(['csdp', problem_file, solution_file])

    # Extract block one of primal solution and do a cholesky
    dual_sol, primal_sol = read_sdp_solution(solution_file)

    X_chol = np.linalg.cholesky(primal_sol[1])

    best_solution = find_best_rounded_solution(image, X_chol, 
                                               rounding_iterations,
                                               lambda_reg,
                                               r_neighbours)

    x,y = image.shape
    reconstructed_image = solution_to_image(best_solution, x, y)
    
    return X_chol, best_solution, reconstructed_image


def extract_sub_images(image, sub_image_size, overlap):
    sub_images = []  
  
    x_dim, y_dim = image.shape
    x_ind = 0
    y_ind = 0

    p = overlap
    s = sub_image_size

    def append_sub_image(xl, xu, yl, yu, i, j):

        # Appand an subimage spanned by image[xl:xu, yl:yu]
        # xu as in x_upper, and xl as x_lower etc.

        # Depending of where we sampled the subimage the usable pixels will be different.
        # Say we want to have a 2 pixel overlap (or border) around 

        # If the subimage is taken from a corner, compared to taken from somewhere in the middle.
        # This code cxtract usable pixels from subimage, depening on its position in the image.
        # Usable part is defined by a rectangle spanned by two points (x0,y0), (x1,y1)

        # Upper Left Courner
        if xl == 0 and yl == 0:
            x0 = 0
            y0 = 0

            x1 = s - p
            y1 = s - p
        # Upper Border            
        elif xl == 0 and yl != 0 and yu != y_dim:
            x0 = 0
            y0 = p

            x1 = s - p
            y1 = s - p
        # Upper Right Corner
        elif xl == 0 and yu == y_dim:
            x0 = 0
            y0 = p

            x1 = s - p
            y1 = s
            #import ipdb; ipdb.set_trace()
        # Left Border
        elif xl != 0 and xu != x_dim and yl == 0:
            x0 = p
            y0 = 0

            x1 = s - p
            y1 = s - p
        # Inner
        elif xl > 0 and yl > 0 and xu < x_dim and yu < y_dim:
            x0 = p
            y0 = p

            x1 = s - p
            y1 = s - p
        # Right border
        elif yu == y_dim and xl != 0 and xu != x_dim:
            x0 = p
            y0 = p

            x1 = s - p
            y1 = s
        # Lower border
        elif xu == x_dim and yl != 0 and yu != y_dim:
            x0 = p
            y0 = p

            x1 = s
            y1 = s - p
        # Lower left corner
        elif xu == x_dim and yl == 0:
            x0 = p
            y0 = 0

            x1 = s
            y1 = s - p
        # Lower right courner
        elif xu == x_dim and yu == y_dim:
            x0 = p
            y0 = p

            x1 = s
            y1 = s
        else:
            # This should not happen..
            raise RuntimeError('This should not happen')

        usable_rect = (x0, y0, x1, y1)
        sub_image = image[xl:xu, yl:yu]
        assert(y1 - y0 > 0 and x1 - x0 > 0)
        sub_images.append((sub_image, (i, j), usable_rect))

    # Lower and and upper bounds for subimages
    xl, xu = 0, s

    done = False
    i = 0
    while True:
        yl, yu = 0, s
        j = 0
        while True:
            # Still inside image?
            if yu < y_dim:
                append_sub_image(xl, xu, yl, yu, i, j)
            elif yu == y_dim:
                append_sub_image(xl, xu, yl, yu, i, j)
                break
            else:
                yu = y_dim
                append_sub_image(xl, xu, yl, yu, i, j)
                break
            yl = yu - 2*p
            yu = yl + s
            j += 1

        if done or xu == x_dim:
            break

        xl = xu - 2*p
        xu = xl + s

        if xu > x_dim:
            xu = x_dim
            done = True

        i += 1

    return sub_images


def merge_subimages(sub_images, original_shape):
    x,y = original_shape

    rows = []

    image_row = None
    current_row = -1
    # First we recreate the rows.
    for image, position, usable_rect in sub_images:
        x0, y0, x1, y1 = usable_rect

        # Check if we changed row
        if current_row != position[0]:
            # If we finished one row, add it to row list
            if image_row is not None:
                rows.append(image_row)
            # Reset image_row to create next row
            image_row = None

        usable_pixels = image[x0:x1, y0:y1]

        if image_row is None:
            image_row = usable_pixels
            current_row = position[0]
        else:
            image_row = np.hstack((image_row, usable_pixels))

    rows.append(image_row)
    # All rows are merged back to one big image.
    merged = np.vstack(rows)
    return merged


def segmented_reconstructions(image, lambda_reg, r, sub_image_size, overlap,
                              problem_prefix,
                              solution_prefix,
                              rounding_iterations):

    # Exract subimages
    sub_images = extract_sub_images(image, sub_image_size, overlap)

    reconstructed_images = []
    # Solve each sub problem
    for sub_image, pos, usable_rect in sub_images:
        _, _, reconstruction = reconstruct_image(sub_image,
                                                 lambda_reg=lambda_reg,
                                                 r=r,
                                                 problem_file='subproblem_{}_{}.sdpa'.format(pos[0], pos[1]),
                                                 solution_file='subproblem_{}_{}.sol'.format(pos[0], pos[1]),
                                                 rounding_iterations=rounding_iterations)

        reconstructed_images.append((reconstruction, pos, usable_rect))

    # Extract usable part of subimages and stitch them together
    restitched_image = merge_subimages(reconstructed_images, image.shape)
    return restitched_image


def extract_blocks(block_tree, block_size):

    matrices = {}
    for block_num in block_tree.keys():
        size = block_size[block_num] + 1
        matrices[block_num] = np.zeros((size, size), np.float64)
        for x,y, val in block_tree[block_num]:
            matrices[block_num][x,y] = val
            matrices[block_num][y,x] = val

    return matrices


def read_sdp_solution(filename):

    # Read the solution from csdp solver from 'filename'

    # Returns the dual and primal block as dictionories,
    # mapping blocknumber to matrix.
    
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

        if prim_dual == 1:
            dual_blocks[block_num].append((x,y,value))
            dual_block_max[block_num] = max(dual_block_max[block_num], x)
        else:
            primal_blocks[block_num].append((x,y,value))
            primal_block_max[block_num] = max(primal_block_max[block_num], x)

    dual_matrices = extract_blocks(dual_blocks, dual_block_max)
    primal_matrices = extract_blocks(primal_blocks, primal_block_max)

    return dual_matrices, primal_matrices
