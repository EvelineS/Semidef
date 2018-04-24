import scipy
import matplotlib.pyplot as plt
import numpy as np
import PIL 
from skimage.color import rgba2rgb
from skimage.color import rgb2gray
import skimage.io
import skimage
from skimage import transform as tf
import sys
import subprocess
import itertools
from parse_sdp_sol import read_sdp_solution

from scipy import ndimage


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec






finger = skimage.io.imread('fingerprint_2.png')

#finger = skimage.io.imread('text.png')
#finger = skimage.io.imread('escher.jpeg')

escher_big = skimage.io.imread('escher_cube.jpg')
escher_big = rgb2gray(escher_big)
escher_big = escher_big

escher_big = tf.resize(escher_big, (80,80), mode='constant')

binary_escher = escher_big
binary_escher[binary_escher > 0.5] = 1
binary_escher[binary_escher <= 0.5] = 0.0

forest = skimage.io.imread('forest.jpg')

forest = rgb2gray(forest) 
forest = tf.resize(forest, (45 , 45), mode='constant')

plt.imshow(forest)
#plt.show()


skimage.io.imsave("binary_escher.jpeg", binary_escher)

print(finger.shape)
finger = rgba2rgb(finger)

finger = rgb2gray(finger)

finger = tf.resize(finger, (45,45), mode='constant')

print(finger.shape)


# Make it a binary image
finger[finger >= 0.5] = 1.0
finger[finger < 0.5] = 0.0



# Blur with a gaussian convolution
blurred_image = ndimage.filters.gaussian_filter(finger, sigma=0.75)

# Adding gaussian noise 
h, w = blurred_image.shape
blurred_noisy = blurred_image + 0.2*np.random.randn(h,w)

# Make sure we're still inside [0,1] even after addning blur + noise
blurred_image = np.clip(blurred_image, a_min=0.0, a_max=1.0)
blurred_noisy = np.clip(blurred_noisy, a_min=0.0, a_max=1.0)


plt.subplot(131)
plt.imshow(finger, cmap="gray", vmax=1.0, vmin=0.0)
plt.subplot(132)
plt.imshow(blurred_image, cmap='gray', vmax=1.0, vmin=0.0)
plt.subplot(133)
plt.imshow(blurred_noisy, cmap='gray', vmax=1.0, vmin=0.0)
#plt.show()

plt.imshow(escher_big, cmap='gray')
#plt.show()
print(np.max(escher_big))
print(np.min(escher_big))
print(escher_big.dtype)


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
                if y_delta < 0  or y_delta >= width:
                    continue
                
                if (x_delta, y_delta) == (x, y):
                    continue
                neighbours.append((x_delta, y_delta))
        return neighbours
    

    for i in range(height):
        for j in range(width):
            #print("Neighbours to ({},{})".format(i,j))
            for neighbour in get_r_neighbours(i,j):
                #print("\t({}, {})".format(neighbour[0], neighbour[1]))
                
                # i ~ j is also j ~ i so I insterted them lexiographicaly sorted.
                # Will make it easier to make then unique.
                
                if (i ,j) < neighbour:
                    adjacent_nodes.append( ( (i, j), neighbour))
                else:
                    adjacent_nodes.append( (neighbour, (i, j)) )
    
    # Make unique
    return set(adjacent_nodes)


def convert_coordinates_to_sdpa(nodes, image):
    """
    We have an image with shape (width, height).
    From the list of r-adjcajent nodes ( (x_i, y_i),  (x_j, y_j) )
    
    We need to flatten the image to a width*heigth length vector.
    Further more we need a index starting from 1 going to N = width*height (inclusive).
    
    This functions maps the adjenct list to the flattened, 1 indexed vector format.
    
    """
    
    height, width = image.shape
    
    def to_flat(x,y):
        return width*x + (y + 1)
    
    return [(to_flat(adjacent[0][0], adjacent[0][1]), 
             to_flat(adjacent[1][0], adjacent[1][1])) for adjacent in nodes]


def sdpa_r_neighbours(r, image):
    nodes = set_of_adjacent_nodes(r, image)
    return convert_coordinates_to_sdpa(nodes, image)


def build_sdpa_problem(image, lambga_reg, r_neighbours, filename):
    
    # Image is assumed to have values in [0, 1]
    # We shift this to [-1, 1]
    
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
        print(X_dim, file=sdpa)
    
        # Number of blocks
        print(1, file=sdpa)
    
        # Block sizes
        print(X_dim, file=sdpa)

        # RHS of constrains, all ones.
        print(" ".join(['1']*X_dim), file=sdpa)

        # LHS contraints X_ii 
        for i in range(1, X_dim + 1):
            print("{} 1 {} {} 1.0".format(i,i,i), file=sdpa)

        # Cost function
        # X_1,2 = flat_image[0]
        # ...
        # X_1,(N+1) = flat_image[N-1]
        for i, g in enumerate(flat_image):
            print("0 1 1 {} {}".format(i+2, g), file=sdpa)
            
        for i,j in r_neighbours:
            if lambga_reg != 0.0:
                print("0 1 {} {} {}".format(i+1, j+1, lambga_reg), file=sdpa)


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
        x_rounded = np.sign(X_chol @ z)

        current_cost = eval_rounded_solution(flat_image, x_rounded, lambda_reg, r_neighbours)
        
        #print("Current cost {}".format(current_cost))
        
        if current_cost > best_cost:
            print("New best: {}".format(current_cost))
            best_cost = current_cost
            best_rounded_solution = x_rounded

    return best_rounded_solution

def eval_rounded_solution(flat_image, rounded_solution, lambda_reg, r_neighbours): 
    
    data_fidelity = 2.0 * flat_image.T @ rounded_solution[1:]
    
    regularization_terms = 0.0
    
    if lambda_reg == 0.0:
        return data_fidelity
    
    for i, j in r_neighbours:
        data_fidelity += rounded_solution[i]*rounded_solution[j]
    
    return data_fidelity + lambda_reg * regularization_terms


def execute(command):
    
    print(command)
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
    print("Building SDP file: {}".format(problem_file))
    
    adj_nodes = set_of_adjacent_nodes(r, image)
    r_neighbours = convert_coordinates_to_sdpa(adj_nodes, image)
    r_neighbours.sort() 
    
    print("Found {} neighbours for r = {}".format(len(r_neighbours), r))
    
    build_sdpa_problem(image, lambga_reg=lambda_reg, 
                       r_neighbours=r_neighbours, filename=problem_file)

    print("Solving SDP")
    #res = subprocess.run(['csdp', problem_file, solution_file], stdout=subprocess.PIPE)

    execute(['csdp', problem_file, solution_file])
        
    
    
    dual_sol, primal_sol = read_sdp_solution(solution_file)
    # Extract block one of primal solution and do a cholesky
    X_chol = np.linalg.cholesky(primal_sol[1])
    #recovered_image = rounding_procedure(X_chol, rouding_iterations, neighbours)
    
   
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
        print("==================")
        print("({},{}),({},{})".format(xl, xu, yl, yu))
        print(image[xl:xu, yl:yu])


        # Extract usable pixels from subimage.
        # Defined by a rectangle spanned by two points (x0,y0), (x1,y1)
        

        # If a subimage is on the border on the image we may 
        
        # Upper Left Courner
        if xl == 0 and yl == 0:
            print("ULC")
            x0 = 0
            y0 = 0

            x1 = s - p
            y1 = s - p
        # Upper Border            
        elif xl == 0 and yl != 0 and yu != y_dim:
            print("UB")
            x0 = 0
            y0 = p

            x1 = s - p
            y1 = s - p
        # Upper Right Corner
        elif xl == 0 and yu == y_dim:
            print("URC")
            x0 = 0
            y0 = p

            x1 = s - p
            y1 = s
            #import ipdb; ipdb.set_trace()
        # Left Border
        elif xl != 0 and xu != x_dim and yl == 0:
            print("LB")
            x0 = p
            y0 = 0

            x1 = s - p
            y1 = s - p

        # Inner
        elif xl > 0 and yl > 0 and xu < x_dim and yu < y_dim:
            print("Inr")
            x0 = p
            y0 = p

            x1 = s - p
            y1 = s - p
        # Right border
        elif yu == y_dim and xl != 0 and xu != x_dim:
            print("RB")
            x0 = p
            y0 = p

            x1 = s - p
            y1 = s
        # Lower border
        elif xu == x_dim and yl != 0 and yu != y_dim:
            print("LowB")
            x0 = p
            y0 = p

            x1 = s
            y1 = s - p
            
        # Lower left corner
        elif xu == x_dim and yl == 0:
            print("LLC")
            x0 = p
            y0 = 0

            x1 = s
            y1 = s - p
        # Lower right courner
        elif xu == x_dim and yu == y_dim:
            print("LRC")
            x0 = p
            y0 = p

            x1 = s
            y1 = s
        else:
            # This should not happen..
            raise RuntimeError('This should not happen')


        usable_rect = (x0, y0, x1, y1)
        sub_image = image[xl:xu, yl:yu]
        print("Usable part")
        print(sub_image[x0:x1, y0:y1])
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
            yl = yu - p - 1
            yu = yl + s
            j += 1

        if done or xu == x_dim:
            break

        xl = xu - p - 1
        xu = xl + s

        if xu > x_dim:
            print("max, xl = {}".format(xl))
            xu = x_dim
            done = True

        i += 1

    return sub_images


def merge_subimages(sub_images, original_shape):
    for image, position, usable_rect in sub_images:
        print("Pos {}".format(position))

    print("Got {}".format(len(sub_images)))
    x,y = original_shape

    rows = []

    image_row = None
    current_row = -1
    for image, position, usable_rect in sub_images:
        print("Pos {}".format(position))
        print(usable_rect)
        x0, y0, x1, y1 = usable_rect
        print(x0, y0, x1, y1)

        # Check if we changed row
        if current_row != position[0]:
            # If we finished one row, add it to row list
            if image_row is not None:
                print("Appending row with shape {}".format(image_row.shape))
                print(image_row)
                rows.append(image_row)
            # Reset image_row to create next row
            image_row = None

        usable_pixels = image[x0:x1,y0:y1]
        print("Sub image")
        print(image)
        print("usable pixels")
        print(usable_pixels)
        
        if image_row is None:
            image_row = usable_pixels
            current_row = position[0]
        else:
            image_row = np.hstack((image_row, usable_pixels))
            print("Created")
            print(image_row)

    print("Appending last row outside loop")
    print(image_row)
    rows.append(image_row)

    merged = np.vstack(rows)
    print("Merged")
    print(merged)
    print(merged.shape)
    return merged


def stitch_subimages(sub_images):

    return None
    
    full_size = sub_images[0][1].shape
    truncated_size = sub_images[-1][1].shape
    
    last_sub_image_positon = sub_images[-1][0]
    
    x_dim = (last_sub_image_positon[0])*(full_size[0] - overlap) + truncated_size[0]
    y_dim = (last_sub_image_positon[1])*(full_size[1] - overlap) + truncated_size[1]
    
    stitched_image = np.zeros((x_dim, y_dim))
    
 
    for coord, sub_image in sub_images:
        x = coord[0]
        y = coord[1]
        
        lower_x = x*(full_size[0] - overlap)
        lower_y = y*(full_size[1] - overlap)
        
        if sub_image.shape[0] == full_size[0]:
            delta_x = full_size[0] - overlap
        else:
            delta_x = truncated_size[0]
            
        if sub_image.shape[1] == full_size[1]:
            delta_y = full_size[1] - overlap
        else:
            delta_y = truncated_size[1]   
               
        stitched_image[lower_x:(lower_x + delta_x), lower_y:(lower_y+delta_y)] = sub_image[0:delta_x,0:delta_y]
    
    
    
    #print(sub_images)
    print(full_size)
    print(truncated_size)
    print(last_sub_image_positon)
    print(stitched_image.shape)
    return stitched_image


def segmented_reconstructions(image, lambda_reg, r, sub_image_size, overlap,
                              problem_prefix,
                              solution_prefix,
                              rounding_iterations):
    
    sub_images = extract_sub_images(image, sub_image_size, overlap)
    
    
    
    # A sanity check to confirm that splitting to subimages and stitching them together gives
    # back the orignal image.
    restitched_image = stitch_subimages(sub_images, overlap)
    assert np.allclose(image, restitched_image)

    #plt.imshow(restitched_image, cmap='gray')
    #plt.show()
    
    reconstructed_images = []
    for coord, sub_image in sub_images:
        
        print("Sub image pos: {}, size = {}".format(coord, sub_image.shape))
        
        _ , _, reconstruction = reconstruct_image(sub_image, lambda_reg=lambda_reg, r=r, 
                                    problem_file='subproblem_{}_{}.sdpa'.format(coord[0], coord[1]),
                                    solution_file='subprolem_{}_{}.sol'.format(coord[0], coord[1]),
                                    rounding_iterations=rounding_iterations)

        to_save = (reconstruction + 1.0)/2.0
        skimage.io.imsave('subimage_reconstructed_ldb{}_{}.jpeg'.format(coord[0], coord[1]), reconstruction)
        reconstructed_images.append((coord, reconstruction))
        
    restitched_image = stitch_subimages(reconstructed_images, overlap)
    return restitched_image

if __name__ == '__main__':


    lambdas = np.linspace(0.01, 0.05, 20)
    #for lbd in lambdas: 
    #    reconstructed = segmented_reconstructions(image=escher_big,
    #                                              lambda_reg=lbd,
    #                                              r=5,
    #                                              sub_image_size=30,                          
    #                                              overlap=3,
    #                                              problem_prefix='escher_big_',
    #                                              solution_prefix='escher_big_',
    #                                              rounding_iterations=100)

    #   reconstructed = (reconstructed + 1.0)/2.0
    #   skimage.io.imsave('escher_reconstructed_ldb{}.jpeg'.format(lbd), reconstructed)


    #plt.imshow(reconstructed, cmap='gray', vmax=1.0, vmin=0.0)

    #plt.show()


    rounding_iterations = 30

    for ldb in lambdas:
        _, _, reconstructed = reconstruct_image(blurred_noisy, lambda_reg=ldb, r=1, 
                                            problem_file='finger.sdpa',
                                            solution_file='finger.sol',
                                            rounding_iterations=rounding_iterations)
        reconstructed = (reconstructed + 1.0)/2.0
        skimage.io.imsave('finger_reconstructed_ldb{}.jpeg'.format(ldb), reconstructed)

        #X_chol_blurred = solve_problem(blurred_image, lambda_reg=0.001, r=1, 
        #                              problem_file='binary_restoration_blurry.sdpa',
        #                              solution_file='binary_restoration_blurry.sol',
        #                              rounding_iterations=rounding_iterations)


        #_, _, blurry_reconstruction = reconstruct_image(blurred_noisy, lambda_reg=0.0, r=1, 
        #                                    problem_file='binary_restoration_blurry_noise.sdpa',
        #                                    solution_file='binary_restoration_blurry_noise.sol',
        #                                    rounding_iterations=rounding_iterations)

        #import ipdb; ipdb.set_trace()
        #plt.imshow(blurry_reconstruction, cmap='gray', vmax=1.0, vmin=-1.0)
        #plt.show()

        print("Done")
