"""
Automatic compilation of ptx files, useful for differential analysis
"""
# W.J. van der Laan 2007
from subprocess import Popen,PIPE
from Disass import load
from tempfile import mkstemp
from os import remove,fdopen,close

ptxas = "ptxas"

class CompilationError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)
        self.message = msg

class PTX:
    def __init__(self, ptx):
        ptxout = mkstemp(suffix=".ptx")
        binout = mkstemp(suffix=".cubin")
        try:
            close(binout[0]) # we are only interested in the name
            # Write ptx
            f = fdopen(ptxout[0], "w")
            f.write(ptx)
            f.close()

            # Compile ptx
            sub = Popen([ptxas,ptxout[1],'-o',binout[1]], stdout=PIPE, stderr=PIPE)
            rv = sub.communicate()
            if sub.returncode:
                raise CompilationError(rv[1])
            
            # Open cubin
            self.bin = load(binout[1])
        finally:
            # Remove temp files
            remove(ptxout[1]) 
            remove(binout[1]) 

