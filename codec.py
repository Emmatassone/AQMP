import numpy as np
from rawfile import RawFile
from omp import OMPHandler
from utility import Utility
from PIL import Image


class ImageCompressor:
    def __init__(self, min_sparcity, min_n, max_n, a_cols, fif_version, magic_number, header_format, v_format_precision, max_error):
        self.min_sparcity = min_sparcity
        self.min_n = min_n
        self.max_n = max_n
        self.a_cols = a_cols
        self.fif_version = fif_version
        self.magic_number = magic_number
        self.header_format = header_format
        self.v_format_precision = v_format_precision
        self.max_error = max_error
        self.omp_handler = OMPHandler(self.min_n, self.max_n, self.a_cols, self.min_sparcity)
        self.omp_handler.initialize_dictionary()

    def code(self, input_file, output_file): # quedaria mejor llamar a la funcion 'encode'?
        """Compress input_file with the given parameters into output_file"""
        image = Image.open(input_file)
        image = image.convert('YCbCr')
        w, h = image.size
        depth = Utility.mode_to_bpp(image.mode) // 8
        self.image_rawsize = w * h * depth

        with RawFile(output_file, 'wb') as file:
            # print(self.header_format, self.magic_number, self.fif_version, w, h, depth, 0, 0, self.min_n, self.max_n,"\n")

            file.write_header(
                self.header_format,
                self.magic_number,
                self.fif_version,
                w,
                h,
                depth,
                0, #A_id (to be removed)
                0, #basis_index (to be removed)
                self.min_n,
                self.max_n
            )

            image_data = np.array(image.getdata()).reshape(h, w, depth)

            processed_blocks = 0
            n = self.max_n
            x_list = []
            for k in range(depth):
                channel_processed_blocks, x_list = self.omp_handler.omp_code(
                                                                        x_list = x_list,
                                                                        image_data = image_data[:, :, k],
                                                                        max_error = self.max_error,
                                                                        block_size = n,
                                                                        k = k,
                                                                        )
                processed_blocks += channel_processed_blocks

            for n, i, j, k, x in x_list:
                file.write("H", i)
                file.write("H", j)
                file.write("B", k)
                file.write("B", n)
                file.write_vector(x.tolist(), self.v_format_precision)
                # print("i, j, k, n: ", i, j, k, n)
            file.write("I", processed_blocks)

            bytes_written = file.tell()
            print(f"bytes_written: {bytes_written}")
            print(f"processed_blocks: {processed_blocks}")

    def decode(self, input_file, output_file):
        """Decompress input_file into output_file"""
        with RawFile(input_file, 'rb') as file:
            file.seek(-4, 2)
            processed_blocks = file.read("I")
            file.seek(0)
            # print("processed_blocks:",processed_blocks)
            #No es necesario ya que la clase guarda estos parámetros
            #A_id, basis_index = 0, 0 (to be removed)
            magic_number_read, version, w, h, depth, A_id, basis_index, min_n, max_n = file.read(self.header_format)
            # print(magic_number_read, version, w, h, depth, A_id, basis_index, min_n, max_n)

            if magic_number_read != self.magic_number.decode('utf-8'):
                raise Exception(f"Invalid image format: Wrong magic number '{magic_number_read}'")

            if version != self.fif_version:
                raise Exception(f"Invalid codec version: {version}. Expected: {self.fif_version}")

            image_data = np.zeros((h, w, depth), dtype = np.float32)
            image_data = self.omp_handler.omp_decode(
                                                      file,
                                                      image_data,
                                                      self.max_n,
                                                      self.v_format_precision,
                                                      processed_blocks
                                                    )

            if depth == 1:
                image_data[:, :, 1] = image_data[:, :, 0]
                image_data[:, :, 2] = image_data[:, :, 0]

            image_data = Utility.ycbcr_to_rgb(image_data)

            image = Image.fromarray(image_data.astype('uint8'))
            image.save(output_file)
            print("Output file saved to: " + output_file)
