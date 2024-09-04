import numpy as np

class Utility:
    @staticmethod
    def truncate(value, format_spec):
        """Truncate the value to the specified format."""
        try:
            return float(f"{value:{format_spec}}")
        except ValueError as error:
            raise ValueError(f"Invalid format code '{format_spec}' for value '{value}'") from error

    @staticmethod
    def quantize(array, v_format):
        """Truncate elements of 'array' using v_format"""
        for elem in array:
            elem = Utility.truncate(elem, v_format)
        return array

    @staticmethod
    def mode_to_bpp(mode):
        """Convert image mode to bits per pixel."""
        if mode == 'L':
            return 8
        elif mode == 'RGB':
            return 24
        elif mode == 'RGBA':
            return 32
        elif mode == 'YCbCr':
            return 24
        else:
            raise ValueError(f"Unsupported image mode: {mode}")

    @staticmethod
    def min_sparcity(max_error, N):
        """Calculate minimum sparsity based on max_error and block size N."""
        return int(np.ceil(max_error * N**2))
    
    @staticmethod
    def sub_image(image_data, n, i, j, k=None):
        """Extract a sub-image from the larger image array."""
        h0, h1, w0, w1 = i, i + n, j, j + n
        if k is None:
            return image_data[h0:h1, w0:w1]
        return image_data[h0:h1, w0:w1, k]

    @staticmethod
    def set_sub_image(sub_img, image_data, n, i, j, k):
        """Place a sub-image back into the larger image array."""
        h0, h1, w0, w1 = i, i + n, j, j + n
        image_data[h0:h1, w0:w1, k] = sub_img

    @staticmethod
    def ycbcr_to_rgb(image_data):
        """
        Reemplazar esta función por la función de la libreria Pillow:
        image = Image.fromarray(image_data, 'YCbCr')
        """
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
    
