import numpy as np
from numpy import array, concatenate, sign
import io
from PIL import Image
from math import pi, cos, sin, log, sqrt
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.sparse import csc_matrix
import zlib
from rawfile import RawFile

min_n = 2**3
max_n = 2**5
a_cols = 256
fif_version = 2
magic_number = b'FIF'  # Ensure this is a bytes object
header_format = '3sBiiBBBBB'
v_format_precision = ".2f"  # Use this for floating-point precision formatting

# por que se usa v_format_precision = ".2f" ?

### encoder ###

def truncate(value, format_spec):
    """Truncate the value to the specified format."""
    try:
        return float(f"{value:{format_spec}}")
    except ValueError as error:
        raise ValueError(f"Invalid format code '{format_spec}' for value '{value}'") from error
        
def quantize(x, v_format):
    """Truncate elements of x using v_format"""
    for elem in x:
        elem = truncate(elem, v_format)
    return x

def _omp_code(x_list, image_data, im_rec, omp_dict, max_error, basis_index, n, k, stats, ssim_stop, min_n, max_n, callback):
    """ Process channel of image using Matching Pursuit. """
    """ The vector of coefficients 'x' is computed. """
    print( f" image_data.flatten().shape {image_data.flatten().shape}")
    y = image_data.flatten()[:1024] # trunco b solamente de prueba para hacer coincidir las dimensiones. arreglar

    A = omp_dict.get(n)
    #print(f"A = {A}")

    if A.shape[1] > y.size:
        A = A[:, :y.size]
    print(f"A.shape: {A.shape}, y.shape{y.shape}")

    # Perform OMP on the entire image
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(A.shape[1], y.size))
    omp.fit(A, y)
    x = omp.coef_

    x_list.append((n, x))
    processed_blocks = 1
    return processed_blocks, x_list

def code(input_file, output_file, max_error, basis_index, min_n=min_n, max_n=max_n):
    """Compress input_file with the given parameters into output_file"""
    print(f"min_n = {min_n}, max_n = {max_n}")

    # STEP 0 - HEADER ###########################
    version = fif_version
    A_id = 0
    ssim_stop = False
    callback = None
    image = Image.open(input_file)
    image = image.convert('YCbCr')
    w, h = image.size
    print(f"image size: {image.size}")
    depth = mode_to_bpp(image.mode) // 8
    raw_size = w * h * depth

    print(f"Image Mode: {image.mode}, Depth: {depth}, Width: {w}, Height: {h}")

    with RawFile(output_file, 'wb') as f:
        print(header_format, magic_number, version, w, h, depth, A_id, basis_index, min_n, max_n,"\n")
        f.write(header_format, magic_number, version, w, h, depth, A_id, basis_index, min_n, max_n)

        # STEP 1 - OMP ###########################
        image_data = np.array(image.getdata()).reshape(h, w, depth)
        stats = {} 
        n0_cumu = 0

        # Initialize dictionary of sparse vectors for the whole image
        omp_dict = {}
        n = max_n

        # while n <= max_n: lo hacemos para un solo 'n'
        A = DCT1_Haar1_qt(n * n, a_cols)
        print(f"Initializing omp_dict[{n}] with matrix A of shape {A.shape}")
        omp_dict[n] = A
        #n *= 2

        x_list = []
        # Process each color channel of the entire image
        for k in range(depth): 
            n0, x_list = _omp_code(x_list = x_list,
                                   image_data = image_data[:, :, k],
                                   im_rec = None,
                                   omp_dict = omp_dict,
                                   max_error = max_error,
                                   basis_index = basis_index,
                                   n = n,
                                   k = k,
                                   stats = stats,
                                   ssim_stop = ssim_stop,
                                   min_n = min_n,
                                   max_n = max_n,
                                   callback = callback
                                   )
            n0_cumu += n0
        print(f"x: {x_list}") # x_list tiene info de los 3 canales
        
        # Write compressed data
        for n, x in x_list:
            # f.write("B", n) # para qué está esta línea?
            write_vector(f, x.tolist(), ".2f")

        bytes_written = f.tell()
        print(f"bytes_written: {bytes_written}")

    return bytes_written, raw_size, n0_cumu

def write_vector(file, x, v_format):
    """Write a sparse vector as a list of pairs (pos, value)"""
    x = quantize(x, v_format)
    x_norm_0 = np.linalg.norm(x, 0)
    print(f"x.shape:  {np.array(x).shape}")
    print(f"norm_0 of x: {x_norm_0}")

    file.write("B", int(x_norm_0))

    position_format = "B" if len(x) <= 256 else "H"
    if x_norm_0 > 0:
        for position, value in enumerate(x):
            if value != 0:
                print(f"position not zero: {position}")
                file.write(position_format, position)
                file.write("f", float(truncate(value, v_format)))

### decoder ###

def read_vector(file):
    """
    Read vector from 'file'
    Read an sparse vector as a list of pairs (pos, value)
    """
    n0 = file.read("B") # ver si esta bien que siempre sea "B"
    x = np.zeros(a_cols)

    print(f"x.shape: {x.shape}")    
    print("\n (pos, value) pairs:\n")

    pos_format = "B" if x.shape[0] <= 256 else "H"
    for _ in range(n0):
        pos = file.read(pos_format)
        value = file.read("f")
        x[pos] = value
        
        print(f"pos: {pos}")
        print(f"value: {value}\n")
    return x

def _omp_decode(file, image_data, basis_index, n, min_n, max_n, v_format):
    """OMP decoder for the entire image"""
    A = DCT1_Haar1_qt(n * n, a_cols)

    # Read the vector x from the file
    x = np.array(read_vector(file))

    # Compute output_vector = A * x
    output_vector = np.dot(A, x)

    for elem in output_vector:
        elem = truncate(elem, "B")

    # Reconstruct the image data (c_inv still not defined. Found in biyections.py)
    image_data[:, :] = c_inv[basis_index](output_vector, n).reshape(image_data.shape)
    
    return image_data

def decode(input_file, output_file):
    """Decompress input_file into output_file"""
    with RawFile(input_file, 'rb') as file:
        #Chequear porque me parece no recuerdo  A_id, basis_index, min_n, max_n si los guardé con el encoder.
        #En particular, la variable  A_id la borré de todo el código
        magic_number_read, version, w, h, depth, A_id, basis_index, min_n, max_n = file.read(header_format)
        print(f"header: {magic_number_read}, {version}, {w}, {h}, {depth}, {A_id}, {basis_index}, {min_n}, {max_n}")

        if magic_number_read != magic_number.decode():
            raise Exception(f"Invalid image format: Wrong magic number '{magic_number_read}'")

        if version != fif_version:
            raise Exception(f"Invalid codec version: {version}. Expected: {fif_version}")

        image_data = np.zeros((h, w, depth), dtype=np.float32)

        # Process image for each channel
        for k in range(depth):
            image_data = _omp_decode(file, image_data[:, :, k], basis_index, max_n, min_n, max_n, v_format_precision)

        if depth == 1:
            image_data[:, :, 1] = image_data[:, :, 0]
            image_data[:, :, 2] = image_data[:, :, 0]

        image_data = YCbCr_to_RGB(image_data)

        image = Image.fromarray(image_data.astype('uint8'))
        image.save(output_file)

###

def YCbCr_to_RGB(image_data):
    """Convert YCbCr to RGB."""
    Y = image_data[:, :, 0]
    Cb = image_data[:, :, 1] - 128
    Cr = image_data[:, :, 2] - 128

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    image_data[:, :, 0] = np.clip(R, 0, 255)
    image_data[:, :, 1] = np.clip(G, 0, 255)
    image_data[:, :, 2] = np.clip(B, 0, 255)
    
    return image_data
    
def sub_image(image_data, n, i, j, k):
    """
    Extracts a sub-image from a larger image array.

    Parameters:
    - image_data: The full image data as a NumPy array.
    - n: The size of the block.
    - i, j: The starting indices for the block.
    - k: The channel index.

    Returns:
    - The extracted sub-image as a NumPy array.
    """
    h0, h1, w0, w1 = i, i + n, j, j + n
    return image_data[h0:h1, w0:w1, k]

def set_sub_image(sub_img, image_data, n, i, j, k):
    """
    Places a sub-image back into the larger image array.

    Parameters:
    - sub_img: The sub-image to be placed back.
    - image_data: The full image data as a NumPy array.
    - n: The size of the block.
    - i, j: The starting indices for the block.
    - k: The channel index.
    """
    h0, h1, w0, w1 = i, i + n, j, j + n
    image_data[h0:h1, w0:w1, k] = sub_img

def mode_to_bpp(mode):
    """Convert image mode to bits per pixel."""
    if mode == 'L':  # 8-bit pixels, black and white
        return 8
    elif mode == 'RGB':  # 24-bit color
        return 24
    elif mode == 'RGBA':  # 32-bit color with alpha
        return 32
    elif mode == 'YCbCr':  # 24-bit color (YUV)
        return 24
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
        
def min_sparcity(max_error, N):
    """Calculate minimum sparsity based on max_error and block size N."""
    return int(np.ceil(max_error * N**2))

def print_progress(message, processed, total):
    """Print progress message."""
    print(message % (processed, total))

def get_progress(stats, image_data, min_n):
    """Get progress information for print."""
    total_blocks = (image_data.shape[0] // min_n) * (image_data.shape[1] // min_n)
    processed_blocks = sum(stats.values())
    return processed_blocks, total_blocks

### Haar basis ###

def phi(x):
	if 0 <= x < 1:
		return 1
	return 0

def psi(x):
	if 0 <= x < 0.5:
		return 1
	if 0.5 <= x < 1:
		return -1
	return 0

def h(i, N):
    """Generate the h function for the Haar basis."""
    if i == 0:
        return phi

    n, k = [(n, k) for n in range(int(log(N, 2))) for k in range(2 ** n)][i - 1]
    return lambda x: 2 ** (n / 2.0) * psi(2 ** n * x - k)

def v(h, N):
    """Generate the v vector for the Haar basis."""
    return [h(i / float(N)) for i in range(N)]

def Haar1_qt(rows, cols):
    """Generate the Haar basis matrix."""
    return np.array([v(h(i, cols), rows) for i in range(cols)]).T

### DCT basis ###

def DCT_II_f(k, N):
    """Discrete Cosine Transform Type II"""
    def f(x):
        return np.cos(pi * (x + 0.5) * k / N)
    return f

def w(k, N):
	c = sqrt(2) ** sign(k)
	return  [ c * DCT_II_f(k, N)(i / float(N)) for i in range(N) ]

def DCT1_qt(rows, cols):
    """Generate the DCT basis matrix."""
    return np.array([w(k, rows) for k in range(cols)]).T

### Combine DCT and Haar ###

def DCT1_Haar1_qt(rows, cols):
    """Combine DCT and Haar basis matrices."""
    dct_matrix = DCT1_qt(rows, cols // 2)
    haar_matrix = Haar1_qt(rows, cols // 2)
    return np.concatenate((dct_matrix, haar_matrix), axis=1)

###

def W(k1, k2, n, N):
    """Generate the W matrix for the basis."""
    def ro(t):
        return [[cos(t), -sin(t)], [sin(t), cos(t)]]

    def theta(n, N):
        return pi * n / (2.0 * N)

    def g(k1, k2, N1, N2, v):
        return DCT_II_f(k1, N1)(v[0]) * DCT_II_f(k2, N2)(v[1])

    def W_elem(i, j):
        return g(k1, k2, 8, 8, np.dot(ro(theta(n, N)), [[i / 8.0], [j / 8.0]]))

    return [[W_elem(i, j) for j in range(8)] for i in range(8)]