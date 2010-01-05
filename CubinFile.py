#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan, 2007
from cStringIO import StringIO
from Operand import *
from Instruction import *
from FlowControl import *
from Constants import *
from Opcodes import *
from Formatter import *
from Disass import Disassembler
from Exceptions import *

class CubinFile(object):
    kernels = None
    architecture = None
    abiversion = None
    modname = None
    kernels_byname = None
    
    def __init__(self):
        self.kernels = []
        self.kernels_byname = {}

    def write(self, f):
        """Write cubin data to f"""
        # Write intro
        # Write constant segments
        f.write("architecture {sm_10}\n")
        f.write("abiversion {0}\n")
        f.write("modname {cubin}\n")
        # test zone 0
        #f.write("consts  {\n")
        #f.write("\tname = ww\n")
        #f.write("\tsegname = const\n")
        #f.write("\tsegnum = 0\n")
        #f.write("\toffset = 0\n")
        #f.write("\tbytes = 4\n")
        #f.write("\tmem  {\n")
        #f.write("\t\t0x12345678 0x20000010 0x20000020 0x20000030\n") # same as for code
        #f.write("\t}\n")
        #f.write("}\n")

        for kernel in self.kernels:
            f.write("code  {\n")
            f.write("\tname = %s\n" % kernel.name)
            f.write("\tlmem = %i\n" % kernel.lmem)
            f.write("\tsmem = %i\n" % kernel.smem)
            f.write("\treg = %i\n" % kernel.reg)
            f.write("\tbar = %i\n" % kernel.bar)
            f.write("\tbincode  {\n")
            # kernel.bincode
            # up to four 32 bit values per line
            for i in xrange(0, len(kernel.bincode), 4):
                f.write("\t\t"+("".join(["0x%08x " % x for x in kernel.bincode[i:i+4]]))+"\n")
            f.write("\t}\n")
            # XXX write local constant stuff
            # test zone 1
            #f.write("\tconst  {\n")
            #f.write("\t\tsegname = const\n")
            #f.write("\t\tsegnum = 1\n")
            #f.write("\t\toffset = 0\n")
            #f.write("\t\tbytes = 4\n")
            #f.write("\t\tmem  {\n")
            #f.write("\t\t\t0x56789abc 0x10000010 0x10000020 0x10000030\n") # same as for code
            #f.write("\t\t}\n")
            #f.write("\t}\n")

            f.write("}\n")

class Label(object):
    name = None
    inst = None
    addr = None
    def __init__(self, name):
        self.name = name
        
    def assemble(self):
        # Label has no instruction representation
        self.inst = []
        
    def __repr__(self):
        return self.name + ":"

class Kernel(object):
    name = None
    lmem = None # Amount of local mem used
    smem = None # Amount of shared mem used
    reg = None  # Number of registers
    bar = None  # Number of barriers
    bincode = None
    const = None
    instructions = None # Disassembled kernel
    
    def __init__(self):
        self.const = []
    
    def __repr__(self):
        rv = StringIO()
        self.disassemble(rv)
        return rv.getvalue()

    def disassemble(self, rv, formatter=Formatter()):
        """Disassemble the cubin instructions in this kernel"""
        # Phase 1 -- decode instructions
        ptr = 0
        disa = Disassembler()
        instructions = []
        while ptr < len(self.bincode):
            base = ptr*4
            inst = [self.bincode[ptr]]
            ptr += 1
            if inst[0] & 1:
                inst.append(self.bincode[ptr])
                ptr += 1
            instructions.append(disa.decode(base, inst))

        # Phase 2 -- labels, sort in order of address
        label_set = set()
        for i in instructions:
            for o in i.dst_operands:
                if o.indirection == OP_INDIRECTION_CODE and o.source == OP_SOURCE_IMMEDIATE:
                    label_set.add(o.value)
        labels = list(label_set)
        labels.sort()
        label_map = dict([(l, "label%i" % x) for x,l in enumerate(labels)])

        # Phase 3 -- fill in labels in program arguments
        for i in instructions:
            for o in i.dst_operands:
                if o.indirection == OP_INDIRECTION_CODE and o.source == OP_SOURCE_IMMEDIATE:
                    o.label = label_map[o.value]
                    
        # Phase 4 -- print
        for i in instructions:
            formatter.address(rv, i.address)
            
            formatter.bincode(rv, (" ".join(["%08x" % x for x in i.inst])))
            if i.address in label_map:
                formatter.label(rv, label_map[i.address])
            i.dump(rv, formatter)
            formatter.newline(rv)
        
        # Phase 5 -- print constants
        for seg in self.const:
            formatter.const_hdr(rv, seg.segname, seg.segnum, seg.offset, seg.bytes)
            formatter.const_data(rv, seg.mem)
            
    def assemble(self):
        # Phase 1 -- assemble instructions, fill in addresses
        bincode = []
        label_map = {} 
        for inst in self.instructions:
            inst.addr = len(bincode)*4   # Fill in addresses
            inst.assemble()
            bincode.extend(inst.inst)
            # Create label map
            if isinstance(inst, Label):
                if inst.name in label_map:
                    raise CompilationError(inst.line, "Duplicate label %s" % inst.name)
                label_map[inst.name] = inst
        
        # Phase 2 -- fill in labels
        for inst in self.instructions:
            if isinstance(inst, Instruction):
                dirty = False
                for o in inst.dst_operands:
                    if o.indirection == OP_INDIRECTION_CODE and o.source == OP_SOURCE_IMMEDIATE:
                        try:
                            o.value = label_map[o.label].addr
                        except KeyError:
                            raise CompilationError(inst.line, "Undefined label %s" % o.label)
                        dirty = True
                if dirty:
                    # Relocate
                    inst.assemble()
                    idx = inst.addr>>2
                    bincode[idx:idx+len(inst.inst)] = inst.inst
        
        self.bincode = bincode
        #print self.instructions

class Const(object):
    segname = None
    segnum = None
    offset = None
    bytes = None
    mem = None

class Dummy:
    """Dummy environment that absorbs environments that we are not interested in"""
    def extend(self, x):
        pass

_numeric = ["offset","segnum","bytes","lmem","smem","reg","bar"]

def load(name):
    """Load a cubin binary assembly file"""
    f = open(name, "r")

    ex = CubinFile()
    inside = [ex]
    while True:
        line = f.readline()
        if not line:
            break
        line = line[0:-1]
        if line.strip() == "":
            # Empty line
            continue

        closebrace = line.rfind("}")
        openbrace = line.find("{")
        equalpos = line.find("=")
        if openbrace != -1:
            cmd = line[0:openbrace].strip()
            if closebrace != -1:
                value = line[openbrace+1:closebrace]
                setattr(inside[-1], cmd, value)
            else:
                #print cmd, "open"
                if cmd == "code":
                    kernel = Kernel()
                    inside[-1].kernels.append(kernel)
                    inside.append(kernel)
                elif cmd == "bincode":
                    inst = []
                    inside[-1].bincode = inst
                    inside.append(inst)
                elif cmd == "const":
                    const = Const()
                    inside[-1].const.append(const)
                    inside.append(const)
                elif cmd == "mem":
                    inst = []
                    inside[-1].mem = inst
                    inside.append(inst)
                elif cmd == "consts" or cmd == "sampler" or cmd == "reloc":
                    # Ignore
                    inside.append(Dummy())
                elif cmd == "params_SMEM":
                    # Ignore
                    inside.append(Dummy())
                else:
                    raise ValueError("Invalid environment %s" % cmd)
        elif closebrace != -1:
            #print inside[-1], "closed"
            inside.pop()
        elif equalpos != -1:
            valname = line[0:equalpos].strip()
            valvalue = line[equalpos+1:].strip()
            if valname in _numeric:
                valvalue = int(valvalue)
            setattr(inside[-1], valname, valvalue)
        else:
            # Bincode?
            inst = line.strip().split(" ")
            inst = [int(x,0) for x in inst]
            inside[-1].extend(inst)
            #print "inst", inst

    # Fill in name->kernel map
    for kernel in ex.kernels:
        ex.kernels_byname[kernel.name] = kernel
            
    return ex
       
