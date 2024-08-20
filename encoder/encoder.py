MIN_N = 8
MAX_N = 32

def code(input_file, output_file, max_error, bi, callback=None, ssim_stop=False, min_n=MIN_N, max_n=MAX_N):
    """Compress input_file with the given parameters into output_file"""
    print("N", min_n, max_n)

    # STEP 0 - HEADER ###########################
    version = FIF_VER
    A_id = 0  # FIXME Use BASIS_MAP_ID[A_name]

    im = Image.open(input_file)
    im = im.convert('YCbCr')
    w, h = im.size
    depth = mode_to_bpp(im.mode) // 8
    raw_size = w * h * depth

    print(f"Image Mode: {im.mode}, Depth: {depth}, Width: {w}, Height: {h}")

    with RawFile(output_file, 'wb') as f:
        f.write(HEADER_FMT, MAGIC_NUMBER, version, w, h, depth, A_id, bi, min_n, max_n)

        # STEP 1 - OMP ###########################
        im_data = np.array(im.getdata()).reshape(h, w, depth)
        im_rec = (0, h, 0, w)
        stats = {} 
        n0_cumu = 0

        omp_d = {}
        n = min_n
        while n <= max_n:
            A = DCT1_Haar1_qt(n * n, A_COLS)  
            print(f"Initializing omp_d[{n}] with matrix A of shape {A.shape}")
            omp_d[n] = A 
            n *= 2

        x_list = []

        for k in range(depth): 
            n0 = _omp_code(x_list, im_data, im_rec, omp_d, max_error, bi, 
                           max_n, k, stats, ssim_stop, min_n, max_n, callback)
            n0_cumu += n0

        for N, x in x_list:
            if min_n < max_n:
                f.write("B", N)
            v_fmt = ".2f"  
            write_vector(f, x.tolist(), v_fmt)

        bytes_written = f.tell()

    # STEP 2 - DEFLATE ###########################
    with open(output_file, 'rb') as f:
        compress = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY)
        zdata = compress.compress(f.read())
        zdata += compress.flush()

    z_file = output_file + ".zf"

    with open(z_file, 'wb') as f:
        f.write(zdata)
        z_bytes_written = f.tell()

    return output_file, z_file, bytes_written, z_bytes_written, raw_size, n0_cumu
