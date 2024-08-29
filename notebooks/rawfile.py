from struct import pack, unpack, calcsize

class RawFile:
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
        #print(f"fmt = {fmt}")
        if fmt != "n":
            fmt = "!" + fmt
            size = calcsize(fmt)
            data = self.file.read(size)
            udata = unpack(fmt, data)

            #print(f"size read: {size}")
            #print(f"data read: {data}")
            #print(f"udata read: {udata}")
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