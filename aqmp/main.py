import sys
from .aqmp import ImageCompressor

def main():
    if len(sys.argv) < 2:
        print("Usage: python compressor.py <image_file> --parameters")
        sys.exit(1)

    image_file = sys.argv[1]
    parameters = sys.argv[2:]

    # Default parameters for ImageCompressor
    min_sparcity = 0.1
    min_n = 8
    max_n = 32
    a_cols = 64
    max_error = 0.01
    wavelet_election = "db1"
    shuffle_dictionary = False
    v_format_precision = "f"
    apply_deflate = False

    # Parse parameters
    for param in parameters:
        key, value = param.split('=')
        if key == "min_sparcity":
            min_sparcity = float(value)
        elif key == "min_n":
            min_n = int(value)
        elif key == "max_n":
            max_n = int(value)
        elif key == "a_cols":
            a_cols = int(value)
        elif key == "max_error":
            max_error = float(value)
        elif key == "wavelet_election":
            wavelet_election = value
        elif key == "shuffle_dictionary":
            shuffle_dictionary = value.lower() == "true"
        elif key == "v_format_precision":
            v_format_precision = value
        elif key == "apply_deflate":
            apply_deflate = value.lower() == "true"

    compressor = ImageCompressor(
        min_sparcity,
        min_n,
        max_n,
        a_cols,
        max_error,
        wavelet_election,
        shuffle_dictionary,
        v_format_precision,
        apply_deflate,
    )

    if image_file.endswith(".png"):
        output_file = image_file.replace(".png", ".fif")
        compressor.encode(image_file, output_file)
    elif image_file.endswith(".fif"):
        output_file = image_file.replace(".fif", ".png")
        compressor.decode(image_file, output_file)
    else:
        print("Unsupported file format. Please use .png for encoding or .fif for decoding.")
        sys.exit(1)

if __name__ == "__main__":
    main()