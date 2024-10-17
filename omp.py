import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from utility import Utility
from basis import BasisFunctions
from anytree import Node, RenderTree, PreOrderIter

# omphandler.py nuevo nombre para el archivo?


class OMPHandler:
    """
    Manage encoding and decoding using Orthogonal Matching Pursuit (OMP) and Adaptive Quadtrees (AQ).
    Functions to visualize the tree structure obtained with AQ.
    """

    def __init__(self, min_n, max_n, a_cols, min_sparcity, max_error):
        self.min_n = min_n
        self.max_n = max_n
        self.a_cols = a_cols
        self.min_sparcity = min_sparcity
        self.max_error = max_error
        self.omp_dict = {}
        self.basis_functions = BasisFunctions()
        self.subdivision_tree = None

    def initialize_dictionary(self):
        """Initialize dictionary for OMP"""
        n_aux = self.min_n
        while n_aux <= self.max_n:
            A = self.basis_functions.DCT1_Haar1_qt(n_aux * n_aux, self.a_cols)
            self.omp_dict[n_aux] = A
            # print(f"A.shape: {A.shape}")
            n_aux *= 2

    def initialize_dictionary2(self, wavelet_election="db1", shuffle=False):
        """Initialize dictionary for OMP"""
        n_aux = self.min_n
        while n_aux <= self.max_n:
            A = self.basis_functions.DCT1_wavelet_qt(
                n_aux * n_aux, self.a_cols, wavelet=wavelet_election, shuffle=shuffle
            )
            self.omp_dict[n_aux] = A
            n_aux *= 2

    def omp_encode(self, x_list, image_data, max_error, block_size, k):
        """Process channel of image using Matching Pursuit and track subdivisions."""
        channel_processed_blocks = 0
        self.subdivision_tree = Node("root")

        for i in range(image_data.shape[0] // block_size):
            for j in range(image_data.shape[1] // block_size):
                channel_processed_blocks, x_list = self.omp_encode_recursive(
                    block_size,
                    i * block_size,
                    j * block_size,
                    k,
                    image_data,
                    max_error,
                    x_list,
                    channel_processed_blocks,
                    parent=self.subdivision_tree,
                )
        return channel_processed_blocks, x_list

    def omp_encode_recursive(
        self,
        block_size,
        from_dim0,
        from_dim1,
        k,
        image_data,
        max_error,
        x_list,
        channel_processed_blocks,
        parent,
    ):
        """Recursive OMP code with tree-based subdivision tracking."""
        sub_image_data = Utility.sub_image(image_data, block_size, from_dim0, from_dim1)
        sub_image_data = sub_image_data.flatten()
        dict_ = self.omp_dict.get(block_size)

        # if dict_.shape[1] > sub_image_data.size:
        #     dict_ = dict_[:, :sub_image_data.size]

        # n_nonzero_coefs = int(dict_.shape[1])
        # print("hyperparameter n_nonzero_coefs:", n_nonzero_coefs)
        omp = OrthogonalMatchingPursuit(
            tol=self.max_error, fit_intercept=False
        )  # min(dict_.shape[1], image_data.size)
        omp.fit(dict_, sub_image_data)
        coefs = omp.coef_
        norm_0_coefs = np.linalg.norm(coefs, 0)  # n_nonzero_coefs_ de sklearn
        # print("norm_0_coefs using linalg:", norm_0_coefs)
        # print("n_nonzero_coefs_ with scikit:", omp.n_nonzero_coefs_)
        # print("coefs:", coefs, coefs.shape)
        # assert int(norm_0_coefs) == omp.n_nonzero_coefs_

        current_node = Node(
            f"Block_{block_size}_{from_dim0}_{from_dim1}", parent=parent
        )

        min_sparcity = Utility.min_sparcity(self.min_sparcity, block_size)
        if norm_0_coefs > min_sparcity and block_size > self.min_n:
            for x_init, y_init in [
                (x, y)
                for x in [0, int(block_size / 2)]
                for y in [0, int(block_size / 2)]
            ]:
                channel_processed_blocks, x_list = self.omp_encode_recursive(
                    int(block_size / 2),
                    from_dim0 + x_init,
                    from_dim1 + y_init,
                    k,
                    image_data,
                    max_error,
                    x_list,
                    channel_processed_blocks,
                    parent=current_node,
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
            # print("i,j,k,n", i,j,k,n)
            A = self.omp_dict[n]
            coefs = np.array(file.read_vector(self.a_cols))
            output_vector = np.dot(A, coefs)
            for elem in output_vector:
                elem = Utility.truncate(elem, v_format_precision)
            image_data[i : i + n, j : j + n, k] = output_vector.reshape((n, n))
        return image_data

    #################################################
    # New functions to visualize the tree structure #
    #################################################

    def print_subdivision_tree(self):
        """Helper function to print the subdivision tree."""
        for pre, _, node in RenderTree(self.subdivision_tree):
            print(f"{pre}{node.name}")

    def summarize_subdivisions(self):
        """Summarizes the tree structure: number of subdivisions, leaf nodes, and tree depth."""
        total_nodes = len([node for node in PreOrderIter(self.subdivision_tree)])
        subdivided_nodes = len(
            [
                node
                for node in PreOrderIter(self.subdivision_tree)
                if len(node.children) > 0
            ]
        )
        leaf_nodes = total_nodes - subdivided_nodes
        max_depth = max([node.depth for node in PreOrderIter(self.subdivision_tree)])

        print(f"Total subimages: {total_nodes}")
        print(f"Subimages that performed subdivisions: {subdivided_nodes}")
        print(f"Subimages that did not subdivide (leaf nodes): {leaf_nodes}")
        print(f"Maximum depth of subdivisions: {max_depth}")
