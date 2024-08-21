import numpy as np
from numpy import array, concatenate, sign
import io
from rawfile import RawFile
from math import pi, cos, sin, log, sqrt
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.sparse import csc_matrix
from struct import pack, unpack, calcsize
import zlib
from PIL import Image

min_n = 8
max_n = 32
a_cols = 256
fif_ver = 2
magic_number = b'FIF'  # Ensure this is a bytes object
header_fmt = '3sBiiBBBBB'
v_fmt_precision = ".2f"  # Use this for floating-point precision formatting

## encoder ##

def truncate(value, format_spec):
    """Truncate the value to the specified format."""
    try:
        return float(f"{value:{format_spec}}")
    except ValueError as e:
        raise ValueError(f"Invalid format code '{format_spec}' for value '{value}'") from e

def quantize(x, v_fmt):
    for i in range(len(x)):
        x[i] = truncate(x[i], v_fmt)

def write_vector_as_pairs(f, x, n0, v_fmt):
    """Write a sparse vector as a list of pairs (pos, value)"""
    f.write("B", int(n0))
    pos_fmt = "B" if len(x) <= 256 else "H"
    if n0 > 0:
        for i, value in enumerate(x):
            if value != 0:
                f.write(pos_fmt, i)
                quantized_value = truncate(value, v_fmt)
                f.write("f", float(quantized_value))     

def write_vector(f, x, v_fmt):
    quantize(x, v_fmt)
    n0 = np.linalg.norm(x, 0)
    write_vector_as_pairs(f, x, n0, v_fmt)

def _omp_code(x_list, im_data, im_rec, omp_d, max_error, bi, N, k, stats, ssim_stop, min_n, max_n, callback):
    b = im_data.flatten()[:1024] # trunco b solamente de prueba para hacer coincidir las dimensiones. arreglar

    A = omp_d.get(N)

    if A.shape[1] > b.size:
        A = A[:, :b.size]
    
    # Perform OMP on the entire image
    print(A.shape, b.shape)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(A.shape[1], b.size))
    omp.fit(A, b)
    x = omp.coef_
    
    x_list.append((N, x))
    processed_blocks = 1
    return processed_blocks, x_list

def code(input_file, output_file, max_error, bi, min_n=min_n, max_n=max_n):
    """Compress input_file with the given parameters into output_file"""
    print(f"min_n = {min_n}, max_n = {max_n}")

    # STEP 0 - HEADER ###########################
    version = fif_ver
    A_id = 0
    ssim_stop = False
    callback = None
    im = Image.open(input_file)
    im = im.convert('YCbCr')
    w, h = im.size
    depth = mode_to_bpp(im.mode) // 8
    raw_size = w * h * depth

    print(f"Image Mode: {im.mode}, Depth: {depth}, Width: {w}, Height: {h}")

    with RawFile(output_file, 'wb') as f:
        f.write(header_fmt, magic_number, version, w, h, depth, A_id, bi, min_n, max_n)

        # STEP 1 - OMP ###########################
        im_data = np.array(im.getdata()).reshape(h, w, depth)
        stats = {} 
        n0_cumu = 0

        # Initialize dictionary of sparse vectors for the whole image
        omp_d = {}
        n = min_n
        while n <= max_n:
            print(f"Initializing omp_d[{n}] with matrix A of shape {A.shape}")
            A = DCT1_Haar1_qt(n * n, a_cols)
            omp_d[n] = A
            n *= 2

        x_list = []
        # Process each color channel of the entire image
        for k in range(depth): 
            n0, x_list = _omp_code(x_list, im_data[:, :, k], None, omp_d, max_error, bi, 
                           max_n, k, stats, ssim_stop, min_n, max_n, callback)
            n0_cumu += n0

        # Write compressed data
        for N, x in x_list:
            f.write("B", N)
            v_fmt = ".2f"  
            write_vector(f, x.tolist(), v_fmt)

        bytes_written = f.tell()

    return output_file, bytes_written, raw_size, n0_cumu

## decoder ##

def read_vector_as_pairs(f, v_fmt):
	"""Read an sparse vector as a list of pairs (pos, value)"""

	x = np.zeros(len(v_fmt))
	n0 = f.read("B")
	pos_fmt = "B" if len(v_fmt) <= 256 else "H"

	for i in range(n0):
		pos = f.read(pos_fmt)
		value = f.read(v_fmt[pos])
		x[pos] = float(value)
	return x
    
def read_vector(f, v_fmt):
    """Read vector from file f with format v_fmt"""
    x = read_vector_as_pairs(f, v_fmt)
    quantize_inv(x, v_fmt)
    return x

def _omp_decode(f, im_data, bi, N, minN, maxN):
    """OMP decoder for the entire image"""

    A = DCT1_Haar1_qt(N * N, a_cols)

    v_fmt = v_fmt_precision

    # Read the vector x from the file
    x = np.array(read_vector(f, v_fmt))

    # Compute the output vector b = A * x
    b = np.dot(A, x)

    for l in range(len(b)):
        b[l] = truncate(b[l], "B")

    # Reconstruct the image data ( c_inv still not defined. Found in biyections.py)
    im_data[:, :] = c_inv[bi](b, N).reshape(im_data.shape)

def decode(input_file, output_file):
    """Decompress input_file into output_file"""

    with RawFile(input_file, 'rb') as f:
        #Chequear porque me parece no recuerdo  A_id, bi, minN, maxN si los guardé con el encoder.
        #En particular, la variable  A_id la borré de todo el código
        mn, version, w, h, depth, A_id, bi, minN, maxN = f.read(header_fmt) 

        if mn != magic_number.decode():
            raise Exception(f"Invalid image format: Wrong magic number '{mn}'")

        if version != fif_ver:
            raise Exception(f"Invalid codec version: {version}. Expected: {fif_ver}")

        im_data = np.zeros((h, w, depth), dtype=np.float32)

        # Process image for each channel
        for k in range(depth):
            _omp_decode(f, im_data[:, :, k], bi, maxN, minN, maxN)

        if depth == 1:
            im_data[:, :, 1] = im_data[:, :, 0]
            im_data[:, :, 2] = im_data[:, :, 0]

        YCbCr_to_RGB(im_data)

        im = Image.fromarray(im_data.astype('uint8'))
        im.save(output_file)

##

def YCbCr_to_RGB(im_data):
    """Convert YCbCr to RGB."""
    Y = im_data[:, :, 0]
    Cb = im_data[:, :, 1] - 128
    Cr = im_data[:, :, 2] - 128

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    im_data[:, :, 0] = np.clip(R, 0, 255)
    im_data[:, :, 1] = np.clip(G, 0, 255)
    im_data[:, :, 2] = np.clip(B, 0, 255)
    
def sub_image(im_data, n, i, j, k):
    """
    Extracts a sub-image from a larger image array.

    Parameters:
    - im_data: The full image data as a NumPy array.
    - n: The size of the block.
    - i, j: The starting indices for the block.
    - k: The channel index.

    Returns:
    - The extracted sub-image as a NumPy array.
    """
    h0, h1, w0, w1 = i, i + n, j, j + n
    return im_data[h0:h1, w0:w1, k]

def set_sub_image(sub_img, im_data, n, i, j, k):
    """
    Places a sub-image back into the larger image array.

    Parameters:
    - sub_img: The sub-image to be placed back.
    - im_data: The full image data as a NumPy array.
    - n: The size of the block.
    - i, j: The starting indices for the block.
    - k: The channel index.
    """
    h0, h1, w0, w1 = i, i + n, j, j + n
    im_data[h0:h1, w0:w1, k] = sub_img

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

def get_progress(stats, im_data, min_n):
    """Get progress information for print."""
    total_blocks = (im_data.shape[0] // min_n) * (im_data.shape[1] // min_n)
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