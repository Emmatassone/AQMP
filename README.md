# Adaptive Quadtree Refinement and Matching Pursuit
Código para Adaptive Quadtree Refinement and Matching Pursuit (AQMP)

# Installation

You can install the Image Compressor package via pip:

```sh
pip install aqmp-compressor
```
# Usage
## Command Line
You can run the image compressor from the command line using the following command:
```sh
python main.py name_of_image.png --min_sparcity=0.1 --min_n=8 --max_n=32 --a_cols=64 --max_error=0.01 --wavelet_election=db1 --shuffle_dictionary=False --v_format_precision=f --apply_deflate=False
```
If no parameters are chosen, these are the predefined ones:

- min_sparcity=0.1
- min_n=8
- max_n=32
- a_cols=64
- max_error=0.01
- wavelet_election=db1
- shuffle_dictionary=False
- v_format_precision=f
- apply_deflate=False

## Importing the ImageCompressor Class
You can also use the ImageCompressor class directly in your Python code. Here is an example of how to use it in a Jupyter Notebook:

```py
from aqmp import ImageCompressor

# Initialize the compressor with default parameters
compressor = ImageCompressor(
    min_sparcity=0.1,
    min_n=8,
    max_n=32,
    a_cols=64,
    max_error=0.01,
    wavelet_election="db1",
    shuffle_dictionary=False,
    v_format_precision="f",
    apply_deflate=False
)

# Compress the image
input_file = 'path/to/your/image.png'
output_file = 'path/to/save/compressed_image.fif'
compressor.encode(input_file, output_file)

# Decompress the image
input_file = 'path/to/save/compressed_image.fif'
output_file = 'path/to/save/decompressed_image.png'
compressor.decode(input_file, output_file)
```
This example demonstrates how to compress and decompress an image using the ImageCompressor class.

# Roadmap:

- [x] Compare with JPEG Algorithm
- [x] Fix dicctionary product with subimages
- [x] Print or show different compression levels
- [ ] Cambiar nombre .fif  (Fast Image Format) por otro más sugerente del algoritmo actual
- [ ] Implement optuna optimization
- [ ] Check alternative dictionaries
   - [ ] Adaptive dictionaries: DCT or wavelet bases, K-means clustering-based dictionary.
   - [ ] Non-orthogonal basis functions: Gabor wavelets or curvelets.
- [ ] Test different compression levels with sparsity parameter.
- [x] Implement DEFLATE function from zlib to see check if there are changes in the SSIM index.
- [ ] Extend to video compressive sensing (2nd paper?) [https://www.mdpi.com/2076-3417/12/5/2734]

# Encoder Diagram:

![Algorithm Example](./images/flow_diagram.png)

# Images:

Test images have been taken from [Here](https://sipi.usc.edu/database/database.php?volume=misc). Filenames and description use are the following

4.1.01        Female (NTSC test image)                   256    Color
4.1.02        Couple (NTSC test image)                   256    Color
4.1.03        Female (from Bell Labs?)                   256    Color
4.1.04        Female                                     256    Color
4.1.05        House                                      256    Color
4.1.06        Tree                                       256    Color
4.1.07        Jelly beans                                256    Color
4.1.08        Jelly beans                                256    Color
4.2.01        Splash                                     512    Color
4.2.03        Mandrill (a.k.a. Baboon)                   512    Color
4.2.05        Airplane (F-16)                            512    Color
4.2.06        Sailboat on lake                           512    Color
4.2.07        Peppers                                    512    Color
