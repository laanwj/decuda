#!/usr/bin/python
#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007

from CubinFile import *
from sys import stdout, stderr
import sys, getopt

from version import VERSION

def usage():
    stdout.write("Decuda .cubin disassembler version %s\n" % VERSION)
    stderr.write("W.J. van der Laan <laanwj@gmail.com>, 2007\n")
    stderr.write("\n")
    stderr.write("Usage: decuda [OPTION]... [FILE]...\n")
    stderr.write("\n")
    stderr.write("-p, --plain             Don't colorize output\n")
    stderr.write("-A, --ansi              Colorize output\n")
    stderr.write("-o, --output-file <x>   Write disassembled kernel to file\n")
    stderr.write("-h, --help              Show this help screen\n")
    stderr.write("-k, --kernel-name <x>   Disassembly kernel name <x> of the cubin file\n")
    stderr.write("-n, --kernel-number <x> Disassemble kernel number <x> of the cubin file\n")
       
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "phAk:n:o:", ["help", "output=","ansi","plain","kernel-name=","kernel-number="])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)

    formatter = Formatter()
    kid = None
    num = None
    of = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-p", "--plain"):
            formatter = Formatter()
        if o in ("-A", "--ansi"):
            formatter = AnsiFormatter()
        if o in ("-k", "--kernel-name"):
            kid = a
        if o in ("-n", "--kernel-number"):
            num = a
        if o in ("-o", "--output-file"):
            of = a
    if len(args) != 1:
        usage()
        sys.exit(2)
            
    cu = load(args[0])
    
    if of != None:
        # Write parsable disassembly to output file
        of = open(of, "w")
        formatter = FileOutFormatter()
        for num,kernel in enumerate(cu.kernels):
            formatter.kernel(of, num, kernel.name)
            of.write(".entry "+kernel.name+"\n")
            of.write("{\n")
            of.write(".lmem %i\n" % kernel.lmem)
            of.write(".smem %i\n" % kernel.smem)
            of.write(".reg %i\n" % kernel.reg)
            of.write(".bar %i\n" % kernel.bar)
            kernel.disassemble(of, formatter)
            of.write("}\n")
        of.close()
    elif kid != None or num != None:
        # Find out which of the kernels in the cubin to disassemble
        try:
            if kid != None:
                kernel = cu.kernels_byname[kid]
            else:
                kernel = cu.kernels[int(num)]
        except (LookupError,ValueError):
            stderr.write("The kernel you have specified does not exist in the cubin file\n")
            sys.exit(2)
    
        formatter.kernel(sys.stdout, num, kernel.name)
        kernel.disassemble(sys.stdout, formatter)
    else:
        # Disassemble all
        for num,kernel in enumerate(cu.kernels):
            formatter.kernel(sys.stdout, num, kernel.name)
            kernel.disassemble(sys.stdout, formatter)

if __name__ == "__main__":
    main()

