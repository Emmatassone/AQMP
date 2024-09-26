import numpy as np
from utility import Utility
from struct import pack, unpack, calcsize

class RawFile:
    """
    Class for reading and writing operations on a file
    """
    
    def __init__(self, name, mode):
        """Open file with name and mode"""
        self.file = open(name, mode)
        self.wnibble = None     # nibble pending to write
        self.rnibble = None     # nibble pending to read
        self.queue = []         # other data pending to write

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()

    def write(self, fmt, *args):
        """Pack and write args with format fmt"""
        if fmt != "n":
            if self.wnibble is None:
                fmt = "!" + fmt  # big-endian. handle binary data in a platform-independent way
                args = [a.encode('utf-8') if isinstance(a, str) else a for a in args]
                self.file.write(pack(fmt, *args))
            else:
                self.queue.append((fmt, args))
        else:
            raise ValueError("fmt = n. Not implemented.")

    def read(self, fmt):
        """Read data with format fmt and unpack"""
        if fmt != "n":
            fmt = "!" + fmt
            size = calcsize(fmt)
            data = self.file.read(size)
            udata = unpack(fmt, data)
            return [u.decode('utf-8') if isinstance(u, bytes) else u for u in udata] if len(udata) > 1 else udata[0]
        else:
            raise ValueError("fmt = n. Not implemented.")

    def tell(self):
        """Return the current file position"""
        return self.file.tell()

    def seek(self, offset, whence=0):
        """Move the file pointer to the specified position"""
        return self.file.seek(offset, whence)

    def close(self):
        """Close the file"""
        if self.wnibble is not None:
            self.write("n", 0)
        self.file.close()

    # Transferred methods from RawFileHandler
    def write_header(self, header_format, *args):
        self.write(header_format, *args)

    def write_vector(self, x, x_norm_0, v_format):
        """ Write a sparse vector as a list of pairs (pos, value)"""
        self.write("B", int(x_norm_0))
        position_format = "B" if len(x) <= 256 else "H"
        if x_norm_0 > 0:
            for position, value in enumerate(x):
                if value != 0:
                    self.write(position_format, position)
                    self.write("f", float(Utility.truncate(value, v_format)))

    def read_vector(self, a_cols):
        """
        Read vector from 'file'
        First read the l0 norm,
        Then read an sparse vector as a list of pairs (pos, value)
        This function is complementary to write_vector function
        """
        x_norm_0 = self.read("B")
        x = np.zeros(a_cols)

        pos_format = "B" if x.shape[0] <= 256 else "H"
        for _ in range(x_norm_0):
            pos = self.read(pos_format)
            value = self.read("f")
            x[pos] = value

        return x
