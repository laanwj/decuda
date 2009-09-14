#!/usr/bin/python
#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007

from Assembler import *
from CubinFile import *
from sys import stdout, stderr
import sys, getopt

from version import VERSION

def usage():
    stdout.write("Cudasm .cubin assembler version %s\n" % VERSION)
    stderr.write("W.J. van der Laan <laanwj@gmail.com>, 2007\n")
    stderr.write("\n")
    stderr.write("Usage: cudasm [OPTION]... [FILE]...\n")
    stderr.write("\n")
    stderr.write("-h, --help              Show this help screen\n")
    stderr.write("-k, --kernel-name <x>   Name of the kernel to assemble\n")
    stderr.write("-o, --output-file <x>   Name of output file\n")
       
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hk:o:", ["help", "kernel-name", "output-file"])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)

    kid = "my_kernel"
    outfile = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-k", "--kernel-name"):
            kid = a
        if o in ("-o", "--output-file"):
            outfile = a
    if len(args) != 1:
        usage()
        sys.exit(2)
    infile = args[0]
    if outfile == None:
        stderr.write("No output file specified\n")
        sys.exit(2)
    
    i = open(infile, "r")
    asm = Assembler()
    try:
        asm.assemble(i)
    except CompilationError,e:
        stderr.write(str(e)+"\n")
        sys.exit(1)
    o = open(outfile, "w")
    asm.output.write(o)
    

if __name__ == "__main__":
    main()

