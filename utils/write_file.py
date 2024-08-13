#import struct and needed files

class RawFile:
    def __init__(self, name, mode):
        """Open file with name and mode"""
        self.f = open(name, mode)
        self.wnibble = None     # Nibble pending to write
        self.rnibble = None     # Nibble pending to read 
        self.queue = []         # Other data pending to write

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
                fmt = "!" + fmt  
                args = [a.encode('utf-8') if isinstance(a, str) else a for a in args]
                self.f.write(pack(fmt, *args))
            else:
                self.queue.append((fmt, args))
        else:
            if self.wnibble is None:
                self.wnibble = (int(args[0]) + 8) & 0xf
            else:
                self.wnibble <<= 4
                self.wnibble |= (int(args[0]) + 8) & 0xf
                self.f.write(pack("B", self.wnibble))
                self.wnibble = None

                for fmt, args in self.queue:
                    self.write(fmt, *args)
                self.queue = []

    def read(self, fmt):
        """Read data with format fmt and unpack"""
        if fmt != "n":
            fmt = "!" + fmt 
            data = self.f.read(calcsize(fmt))
            udata = unpack(fmt, data)
            return [u.decode('utf-8') if isinstance(u, bytes) else u for u in udata] if len(udata) > 1 else udata[0]
        else:
            if self.rnibble is not None:
                udata = self.rnibble
                self.rnibble = None
            else: #nibble case
                data = self.f.read(calcsize("B"))
                udata = unpack("B", data)[0]
                self.rnibble = (udata & 0xf) - 8
                udata = (udata >> 4) - 8
            return udata

    def tell(self):
        """Return the current file position"""
        return self.f.tell()

    def seek(self, offset, whence=0):
        """Move the file pointer to the specified position"""
        return self.f.seek(offset, whence)

    def close(self):
        """Close the file"""
        if self.wnibble is not None:
            self.write("n", 0)
        self.f.close()