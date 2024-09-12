import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from utility import Utility
from basis import BasisFunctions

class OMPHandler:
    def __init__(self, min_n, max_n, a_cols, min_sparcity):
        self.min_n = min_n
        self.max_n = max_n
        self.a_cols = a_cols
        self.min_sparcity = min_sparcity
        self.omp_dict = {}
        self.basis_functions = BasisFunctions() 

    def initialize_dictionary(self):
        """Initialize dictionary for OMP"""
        n_aux = self.min_n
        while n_aux <= self.max_n:
            A = self.basis_functions.DCT1_Haar1_qt(n_aux * n_aux, self.a_cols)
            self.omp_dict[n_aux] = A
            n_aux *= 2
    
    def omp_code(self, x_list, image_data, max_error, block_size, k):
        """ Process channel of image using Matching Pursuit. """
        """ The vector of coefficients 'coefs' is computed. """
        channel_processed_blocks = 0
        # falta caso de subimagenes de los bordes, que pueden ser mas chicas de lo que se trata acá
        for i in range(image_data.shape[0] // block_size):
            for j in range(image_data.shape[1] // block_size):
                channel_processed_blocks, x_list = self.omp_code_recursive(
                                                                            block_size,
                                                                            i*block_size,
                                                                            j*block_size,
                                                                            k,
                                                                            image_data,
                                                                            max_error,
                                                                            x_list,
                                                                            channel_processed_blocks
                                                                        )
        return channel_processed_blocks, x_list

    def omp_code_recursive(self, block_size, from_dim0, from_dim1, k, image_data, max_error, x_list, channel_processed_blocks):
        sub_image_data = Utility.sub_image(image_data, block_size, from_dim0, from_dim1)

        sub_image_data = sub_image_data.flatten()
        dict_ = self.omp_dict.get(block_size)

        if dict_.shape[1] > sub_image_data.size: # para que sirve este if?
            dict_ = dict_[:, :sub_image_data.size]

        # n_nonzero_coefs = min(dict_.shape[1], image_data.size) por que?
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(dict_.shape[1], image_data.size), fit_intercept=False)
        omp.fit(dict_, sub_image_data)
        coefs = omp.coef_
        norm_0_coefs = np.linalg.norm(coefs, 0)
        if norm_0_coefs > self.min_sparcity and block_size > self.min_n: # norm0 >= Utility.min_sparcity(max_error, block_size)
            #print("partitioning")
            for x_init, y_init in [(x, y) for x in [0, int(block_size / 2)] for y in [0, int(block_size / 2)]]:
                channel_processed_blocks, x_list = self.omp_code_recursive(
                    int(block_size / 2), from_dim0 + x_init, from_dim1 + y_init, k, image_data,
                    max_error, x_list, channel_processed_blocks
                )
        else:
            channel_processed_blocks += 1
            x_list.append((block_size, from_dim0, from_dim1, k, coefs))

        return channel_processed_blocks, x_list

    def omp_decode(self, file, image_data, v_format_precision, processed_blocks):
        """OMP decoder for the entire channel"""
        for _ in range(processed_blocks):
            i = file.read("H")
            j = file.read("H")
            k = file.read("B")
            n = file.read("B")
            # print("i,j,k,n:", i,j,k,n)

            A = self.omp_dict[n]
            coefs = np.array(file.read_vector(self.a_cols))
            output_vector = np.dot(A, coefs)
            for elem in output_vector:
                elem = Utility.truncate(elem, v_format_precision)
            image_data[i:i+n, j:j+n, k] = output_vector.reshape((n, n))
        return image_data

    # def omp_code_recursive2(self, block_size, from_dim0, from_dim1, k, image_data, max_error, x_list, channel_processed_blocks):
    #     sub_image_data = Utility.sub_image(image_data, block_size, from_dim0, from_dim1)

    #     sub_image_data = sub_image_data.flatten()
    #     dict_ = self.omp_dict.get(block_size)

    #     # New block code
    #     dict_rows = block_size * block_size
    #     dict_cols = self.a_cols

    #     row_start = (from_dim0 // (dict_rows // 2)) * (dict_rows // 2)
    #     row_end = row_start + (dict_rows // 2)
    #     col_start = (from_dim1 // (dict_cols // 2)) * (dict_cols // 2)
    #     col_end = col_start + (dict_cols // 2)

    #     dict_part = dict_[row_start:row_end, col_start:col_end]

    #     # Continues as in old code
    #     if dict_part.shape[1] > sub_image_data.size:
    #         dict_part = dict_part[:, :sub_image_data.size]

    #     omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(dict_part.shape[1], image_data.size), fit_intercept=False)
    #     omp.fit(dict_part, sub_image_data)
    #     coefs = omp.coef_
    #     norm_0_coefs = np.linalg.norm(coefs, 0)
    #     if norm_0_coefs > self.min_sparcity and block_size > self.min_n: 
    #         # Partition the subimage if the norm is above sparsity threshold and block size is still greater than the minimum
    #         for x_init, y_init in [(x, y) for x in [0, int(block_size / 2)] for y in [0, int(block_size / 2)]]:
    #             channel_processed_blocks, x_list = self.omp_code_recursive(
    #                 int(block_size / 2), from_dim0 + x_init, from_dim1 + y_init, k, image_data,
    #                 max_error, x_list, channel_processed_blocks
    #             )
    #     else:
    #         channel_processed_blocks += 1
    #         x_list.append((block_size, from_dim0, from_dim1, k, coefs))

    #     return channel_processed_blocks, x_list
