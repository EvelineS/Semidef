import unittest

import numpy as np

from hw2_binary_reconstruction import extract_sub_images
from hw2_binary_reconstruction import merge_subimages

class BinaryReconstruction(unittest.TestCase):

    def test_image_stitching(self):

        for dim in range(20, 40):
            for sub_image_size in range(10,15):
                if sub_image_size > dim - 4:
                    continue
                for overlap in [2]:
                    print("dim = {}, sub_image_size = {}, overlap = {}".format(dim, sub_image_size, overlap)) 
                    image = np.arange(dim**2).reshape(dim,dim)
                    x,y = image.shape

                    print(image)
                    print(image.shape)
        
                    sub_images = extract_sub_images(image, sub_image_size, overlap)
                    first_subimage_true = image[0:sub_image_size, 0:sub_image_size]
                    first_subimage = sub_images[0][0]
                    last_sub_image_true = image[x - sub_image_size:x, y - sub_image_size:y]
                    last_sub_image = sub_images[-1][0]

                    print("dim = {}, sub_image_size = {}, overlap = {}".format(dim, sub_image_size, overlap)) 
                    self.assertTrue(np.allclose(first_subimage_true, first_subimage))
                    #self.assertTrue(np.allclose(last_sub_image_true, last_sub_image))
        
                    stitched_image = merge_subimages(sub_images, image.shape)

                    self.assertTrue(np.allclose(stitched_image, image))
                    print("OOOK!")



if __name__ == '__main__':
    unittest.main()
