
def wraplist(x):
    """Wrap x in a list if it isn't a list"""
    if isinstance(x, list):
        return x
    else:
        return [x]

def lookup(x, y):
    """
    Look up an entry in an array or hash, and return an
    approciate response if it runs out of bounds or
    there is a hole.
    """
    try:
        if x[y] != None:
            return x[y]
    except LookupError,e:
        pass
    return "?%i?" % y

def numlookup(x, y):
    """
    Try to resolve y as a numberic, then try to
    look it up a key in dictionary x.
    """
    try:
        return int(y)
    except ValueError:
        return x[y]

def f2i(x):
    """
    float32 x to four-byte representation to uint32
    """
    from struct import pack,unpack
    (x,) = unpack("I", pack("f", x))
    return x

def i2f(x):
    """
    int x to four-byte representation to float32
    """
    from struct import pack,unpack
    (x,) = unpack("f", pack("I", x))
    return x
