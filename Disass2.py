#!/usr/bin/python
# sm1_1 (G80) disassembler version 0.1 (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007

# Old, quirky disassembler
from cStringIO import StringIO

class CubinFile(object):
    kernels = None
    architecture = None
    abiversion = None
    modname = None
    
    def __init__(self):
        self.kernels = []

# sm1_1 atomic operations
atomic_ops = [
"iadd","exch","cas","fadd","inc","dec","imax","imin","fmax",
"fmin","and","or","xor","????","????","????"
]

# memory operand sizes
msize = [".u8",".s8",".u16",".s16",".64",".128",".u32",".s32"]

# logic operations (set)
logic_ops = [None,".lt",".eq",".le",".gt",".ne",".ge",None]

# op d subop 0 operations
d0_ops = [
"and","or","xor",None,None,None,None,None,
None,None,None,"not"
]

# parameter space
param_space = {
0x20: "%gridflags", # lower u16 is gridid
0x21: "%ntid.x",    # checked
0x22: "%ntid.y",
0x23: "%ntid.z",
0x24: "%nctaid.x",
0x25: "%nctaid.y",
0x26: "%ctaid.x",
0x27: "%ctaid.y",
0x28: "%ctaid.z",  # extrapolated
0x29: "%nctaid.y"  # ptx ISA
# 0x60  start of "shared memory" as seen from app
# 0x64  start of actual program parameters
# What are the rest?
}

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

class Kernel(object):
    name = None
    lmem = None # Amount of local mem used
    smem = None # Amount of shared mem used
    reg = None  # Number of registers
    bar = None  # Number of barriers
    bincode = None
    
    def print_oper(self, rv, value, type):
        if type == "r":
            rv.write("$r%02x" % (value))
        elif type == "pr":
            rv.write("[$r%02x]" % (value))
        elif type == "pro":
            rv.write("[ofs+$r%02x]" % (value))
        elif type == "pi":
            rv.write("[%08x]" % (value))
        elif type == "i":
            rv.write("0x%08x" % (value))
        elif type == "y": # parameter
            try:
                rv.write(param_space[value])
            except LookupError:
                rv.write("%%%02x" % (value))
        elif type == "yo": # offset+parameter
            rv.write("%%(ofs+%02x)" % (value))
        elif type == "x": # constant (segment 0, used for global data)
            rv.write("$c%02x" % (value))
        elif type == "z": # constant (segment 1, used for segment specific data)
            rv.write("$d%02x" % (value))
        else:
            rv.write("%02x" % (value))
            
    def print_psize(self, rv, inst, fullinst):
        if fullinst:
            rv.write(".")
            if (inst[1] & 0x08000000):
                rv.write("s")
            else:
                rv.write("u")
            if (inst[1] & 0x04000000):
                rv.write("32")
            else:
                rv.write("16")
        else:
            rv.write(".")
            if (inst[0] & 0x00008000):
                rv.write("b32")
            else:
                rv.write("b16")

    def print_psize2(self, rv, inst, fullinst):
        # xx fullinst and print_psize2
        if fullinst:
            x = inst[1] & 0x00008000
        else:
            x = inst[0] & 0x00008000
        rv.write(".")
        if x:
            rv.write("s32")
        else:
            rv.write("u32")

    
    def disassemble(self):
        """Disassemble the cubin instructions in this kernel"""
        rv = StringIO()
        ptr = 0
        while ptr < len(self.bincode):
            base = ptr*4
            inst = [self.bincode[ptr]]
            ptr += 1
            if inst[0] & 1:
                inst.append(self.bincode[ptr])
                ptr += 1
            rv.write("%04x: %-17s " % (base, " ".join(["%08x" % x for x in inst])))
            op    = (inst[0]&0xF0000000)>>28
            # Misc flags
            flags = (inst[0]&0x0FC00000)>>22
            if len(inst)==1:
                subop = 0
                twoword = False
            else:
                subop = (inst[1]&0xE0000000)>>29
                twoword = True
            
            if len(inst)==2 and (inst[1] & 3)==3:
                immediate = True
            else:
                immediate = False
            if inst[0] & 2: 
                system = True # flow control
            else:
                system = False
            # Is the second instruction present?
            fullinst = twoword and not immediate
            if fullinst: # subsubop
                lop = (inst[1]&0x001fc000)>>14
            else:
                lop = -1
            # Changes some instructions
            # Seems to choose an alternative instruction set, for example,
            # add becomes sub. It is also used on mul24.lo.s32 sometimes
            # but I have no idea what its effect is there.
            alt = flags&1
            
            print_msize = False
            print_psize = False
            print_psize2 = False
            swap_oper = False
            type_float = False
            is_bra = False
            oper1_t = "r" # Operand type u(unknown), r(reg), i(imm), p(pointer)
            oper2_t = "r"
            oper3_t = "r"
            oper4_t = None
            
            # Predication
            if fullinst:
                onpred = (inst[1]&0x00000780)>>7
                pred = (inst[1]&0x00003000)>>12
                if onpred == 15 or onpred == 0:
                    pass # No pred
                elif onpred == 2:
                    # Execute on false
                    rv.write("@!p%i " % pred)
                elif onpred == 5:
                    # Execute on true
                    rv.write("@p%i " % pred)
                else:
                    # ??
                    rv.write("@?%i?p%i " % (onpred,pred))
            
            # Print the main instruction
            if system:
                oper1_t = None
                oper2_t = None
                oper3_t = None
                oper4_t = None
                flags = 0
                alt = 0

                # Special instructions
                if op == 0x1:
                    rv.write("bra")
                    is_bra = True
                elif op == 0x2:
                    rv.write("call")
                    is_bra = True
                elif op == 0x3:
                    rv.write("return")
                elif op == 0x8:
                    rv.write("bar.sync") 
                    # 0x0003xfff where x is the barrier id, 0-15
                    # The other bits are always like this...
                elif op == 0x9:
                    rv.write("trap")
                elif op == 0xA:
                    rv.write("join")  # join point for divergent threads
                    is_bra = True
                else:
                    rv.write("?%i?" % op)
            elif op == 0x0:
                if subop == 0x3: # Load internal value
                    rv.write("ldgpu")
                    if lop == 0:
                        rv.write(".physid")
                    elif lop == 1:
                        rv.write(".clock")
                    elif lop >= 4 and lop < 8:
                        rv.write(".pm%i" % (lop-4))
                    else:
                        rv.write(".??")
                elif subop == 0x6: 
                    rv.write("ld.offset1.shl")
                    oper3_t = "i"
                elif subop == 0x7: 
                    rv.write("st.shared")
                    # TODO: oper1 extends into oper3 for large values
                    oper4_t = "r"
                    oper3_t = None
                    # long offset in word 1
                    # 8 bit reads offset is *4
                    # 16 bit read offset is *2
                    # 32 bit reads offset is *1
                    # always loading at offset 60
                    type = (inst[1]&0x07E00000) >> 21
                    if type == 0:
                        rv.write(".b16")
                    elif type == 33:
                        rv.write(".b32")
                    elif type == 2:
                        rv.write(".b8")
                    else:
                        rv.write(".?%i?" % type)
                    if lop == 0:
                        # Absolute offset
                        rv.write(".abs")
                    elif lop == 1:
                        # Register offset
                        if offsetr: # Use offset register
                            oper2_t = "pro"
                        else:
                            oper2_t = "pr"
                    oper4_t = None
                    #print offsetr
                    #flags = 0 # flags contain something else?
                else:
                    rv.write("????")
            elif op == 0x1:
                print_psize = True
                if subop == 0x0:
                    if lop == 0xF or lop == -1:
                        rv.write("ld")
                    
                        if immediate:
                            oper1_t = None
                        else:
                            oper2_t = "r"
                            oper3_t = None
                    else:
                        rv.write("?%i?" % lop)
                    
                elif subop == 0x1:
                    rv.write("ld.const")  # Load const from offset into register
                    # TODO: oper1 extends into oper3 for large values
                    # TODO: use bit inst[1]&0x00400000 to determine segment to load from
                    if (inst[1]&0x00400000):
                        rv.write("1")
                    else:
                        rv.write("0")
                    oper4_t = None
                    oper3_t = None
                    oper1_t = "pi"
                else:
                    rv.write("????")
            elif op == 0x2:
                print_psize = True
                if subop == 0x0:
                    if alt:
                        alt = False # reset alt flag, as we know the effect in this case
                        rv.write("sub")
                    else:
                        rv.write("add")
                    if fullinst and (inst[1]&0x08000000):
                        # .s32 also signifies saturation
                        rv.write(".sat")
                    if fullinst:
                        oper4_t = "r"
                        oper3_t = None
                else:
                    rv.write("????")
            elif op == 0x3:
                print_psize = True
                
                if subop == 0: # seems to be a sub with arguments swapped
                    rv.write("neg")
                    if fullinst: # for some reason, the arguments for this instruction are oper1,oper2,oper4
                        oper3_t = None
                        oper4_t = "r"
                elif subop == 3:
                    rv.write("set")
                    rv.write(lookup(logic_ops, lop))
                elif subop == 4:
                    rv.write("max")
                elif subop == 5:
                    rv.write("min")
                elif subop == 6:
                    rv.write("shl")
                elif subop == 7:
                    rv.write("shr")
                else:
                    rv.write("????")
            elif op == 0x4:
                #print_psize = True
                if subop == 0:
                    rv.write("mul24")  # XXX hi/lo word
                    print_psize2 = True # based on bit 1 of lop
                    if lop == 0:
                        print_psize2 = False
                        rv.write(".u32.u16")
                    elif lop == 4 or lop == 6 or lop == -1: # no lop specified
                        rv.write(".lo")
                    elif lop == 5 or lop == 7:
                        rv.write(".hi")
                    else:
                        rv.write(".?%i?" % lop)
                    
                else:
                    rv.write("????")
            elif op == 0x5:
                if subop == 0:
                    rv.write("sad") # Sum of Absolute Differences
                    oper4_t = "r"
                    print_psize = True
                else:
                    rv.write("????")
            elif op == 0x6:
                if subop == 0:
                    rv.write("mad24.u32.u16")
                    oper4_t = "r"
                elif subop == 3:
                    rv.write("mad24.lo.u32")
                    oper4_t = "r"
                elif subop == 4:
                    rv.write("mad24.lo.s32")
                    oper4_t = "r"
                elif subop == 5:
                    rv.write("mad24.lo.sat.s32")
                    oper4_t = "r"
                elif subop == 6:
                    rv.write("mad24.hi.u32")
                    oper4_t = "r"
                elif subop == 7:
                    rv.write("mad24.hi.s32")
                    oper4_t = "r"
                else:
                    rv.write("????")
                # mad24.hi.sat.s32 runs over to op==0x7 subop==0x0, but I assume 
                # this is a ptxas bug, as this instruction makes no sense.
            elif op == 0x9:
                type_float = True
                if subop == 0:
                    rv.write("rcp")
                elif subop == 2:
                    rv.write("rsqrt")
                elif subop == 3:
                    rv.write("lg2")
                elif subop == 4:
                    rv.write("sin")
                elif subop == 5:
                    rv.write("cos")
                elif subop == 6:
                    rv.write("ex2")
                else:
                    rv.write("????")
            elif op == 0xa: # Conversion ops
                oper3_t = None
                if subop == 0:
                    rv.write("cvt")
                    types = [".u16",".u32",".u8","??",".s16",".s32",".s8","??"]
                    self.print_psize(rv, inst, fullinst)
                    rv.write(types[lop&7])
                elif subop == 1:  # Used in div implementation
                    rv.write("????")
                elif subop == 2:
                    rv.write("cvt")
                    rops = [".rn",".rm",".rp",".rz"]
                    types = [".u16",".u32",".u8","??",".s16",".s32",".s8","??"]
                    rv.write(rops[lop>>3])
                    rv.write(".f32")
                    rv.write(types[lop&7])
                elif subop == 3: # saturate or scale?
                    rv.write("sat.f32.u32") # conversion; int 0xFFFFFFFF to float 1.0f
                elif subop == 4:
                    rv.write("cvt")
                    rops = [".rni",".rmi",".rpi",".rzi"]
                    if (lop&7)!=1:
                        rv.write("?") # Unknown conversion
                    rv.write(rops[lop>>3])
                    self.print_psize(rv, inst, fullinst)
                    rv.write(".f32")
                elif subop == 6:
                    rops = [".rn",".rm",".rp",".rz"]
                    if (lop&7)!=1:
                        rv.write("?") # Unknown conversion
                    rv.write("cvt")
                    rv.write(lookup(rops, (lop>>3)&3))
                    if (inst[1]&0x08000000):
                        rv.write("i") # integer rounding mode
                    if (lop&32):
                        rv.write(".sat")
                    if (lop&64):
                        rv.write(".abs")
                    rv.write(".f32.f32")
                    
                elif subop == 7:
                    if lop == 1:
                        rv.write("neg")
                        type_float = True
                    else:
                        rv.write("????")
                else:
                    rv.write("????")
            elif op == 0xb:
                type_float = True
                if subop == 0:
                    # This determines the size of the operand (16 or 32 bit) for
                    # integer instructions. For floating point instructions it 
                    # has a different meaning.
                    if fullinst:
                        sizebit = (inst[1]&0x08000000)
                    else:
                        sizebit = (inst[0]&0x00008000)
                    if sizebit and alt:
                        rv.write("????") # an add too?
                    elif alt:
                        rv.write("sub") # d,a,b -> d = a-b
                    elif sizebit:
                        rv.write("sub2") # arguments reversed d,a,b -> d = b-a
                    else:
                        rv.write("add")
                    if fullinst: # for some reason, the arguments for this instruction are oper1,oper2,oper4
                        oper3_t = None
                        oper4_t = "r" 
                    #oper3_t = "i"
                    alt = False
                elif subop == 3:
                    rv.write("set")
                    rv.write(lookup(logic_ops, lop))
                elif subop == 4:
                    rv.write("max")
                elif subop == 5:
                    rv.write("min")
                elif subop == 6: # I don't know what this operation does, but is executed before sin, cos, and lg2
                    # Denormalize maybe?
                    rv.write("presin")
                    oper3_t = None
                else:
                    rv.write("????")
            elif op == 0xc:
                if subop == 0:
                    type_float = True
                    rv.write("mul")
                    if lop == 0 or lop == -1:
                        rv.write(".rn")
                    elif lop == 3:
                        rv.write(".rz")
                    else:
                        rv.write(".??")
                elif subop == 2:
                    oper4_t = "r"
                    rv.write("slct") # Select one of both arguments based on value of oper4
                else:
                    rv.write("????")
            elif op == 0xd:
                if subop == 0x0:
                    ssop = (inst[1]&0x0000C000)>>14
                    print_psize = True
                    
                    rv.write(lookup(d0_ops,lop))
                elif subop == 0x1:
                    rv.write("ld.offset0")
                    # xx inst[0] contains the offset in the area oper1 and oper3 generally are
                    # oper5?
                elif subop == 0x2:
                    print_msize = True
                    oper1_t = "pr"
                    oper2_t = "r"
                    oper3_t = None
                    rv.write("ld.local")
                elif subop == 0x3:
                    print_msize = True
                    swap_oper = True
                    oper1_t = "pr"
                    oper2_t = "r"
                    oper3_t = None
                    rv.write("st.local")
                elif subop == 0x4:
                    print_msize = True
                    oper1_t = "pr"
                    oper2_t = "r"
                    oper3_t = None
                    rv.write("ld.global")
                elif subop == 0x5:
                    print_msize = True
                    oper1_t = "pr"
                    oper2_t = "r"
                    oper3_t = None
                    swap_oper = True # swap src and dest
                    rv.write("st.global")
                elif subop == 0x6: # atomic, ignore output
                    rv.write("atom.global.")
                    rv.write(lookup(atomic_ops,(inst[1]&0x0000003C)>>2))
                    print_msize = True
                elif subop == 0x7: # atomic, provide out reg
                    rv.write("atom.global.out.")
                    rv.write(lookup(atomic_ops,(inst[1]&0x0000003C)>>2))
                    print_msize = True
                else:
                    rv.write("????")
            elif op == 0xe:
                if subop == 0x0:
                    rv.write("mad")
                    type_float = True
                    oper4_t = "r"
                else:
                    rv.write("????")
            elif op == 0xf:
                if subop == 0x0:
                    rv.write("tex")
                    op = (inst[0]&0x03C00000)>>22
                    op2 = (inst[0]&0x04000000)
                    # 0 is also 1d, 1 is also 2d ...
                    if op == 0x8: # tex type
                        rv.write(".1d.f32.f32")
                    elif op == 0x9:
                        rv.write(".2d.f32.f32")
                    elif op == 0xa:
                        rv.write(".3d.f32.f32")
                    elif op == 0xc: 
                        rv.write(".1d.f32.s32")
                    elif op == 0xe:
                        rv.write(".2d.f32.s32")
                    elif op == 0xf:
                        rv.write(".3d.f32.s32")
                    else:
                        rv.write("?%i?" % op)
                    if(op2): # Don't know what this bit does
                        rv.write(".?")
                    flags = 0 # flags are useless here as the bits contain something else
                    alt = 0
                elif subop == 0x7:
                    rv.write("nop")
                else:
                    rv.write("????")
            else:
                # unknown ops: 0x7, 0x8
                rv.write("[op %01x subop %01x]" % (op,subop))

            if fullinst and print_msize: # has second word, not immediate
                opsize = (inst[1] & 0x00E00000)>>21
                rv.write(msize[opsize])
            # size and sign of data, if not immediate
            if print_psize:
                self.print_psize(rv, inst, fullinst)
            if print_psize2:
                self.print_psize2(rv, inst, fullinst)
            if type_float:
                rv.write(".f32")
                
            rv.write(" ")
            if alt:
                rv.write("[alt] ") 

            # Extract and print operands
            if oper3_t != None and (flags&2):
                # operand 3 comes from constant (in segment 0)
                oper3_t = "z"
            if fullinst and (inst[1]&0x00400000):
                #rv.write("[co3] ")
                # operand 4 comes from constant in segment 1
                if oper4_t != None:
                    oper4_t = "x"
                elif oper3_t != None:
                    oper3_t = "x"
            
            #if flags&4:
            #    # operand 1 comes from param
            #    #rv.write("[src from param?] ")
            #    oper1_t = "y" # from param
            #    oper1 += 0x30
            # ld.offset flags 10 means:
            #    multiply src with 4
            #    offset from 
            if (flags&(~(4|2|1))):
                rv.write("[flags 0x%02x] " % flags)
            if fullinst: # Depending on this, source is 7 or 6 bits
                oper1 = (inst[0]&0x0000FE00)>>9
                oper2 = (inst[0]&0x000001FC)>>2
            else:
                oper1 = (inst[0]&0x00007E00)>>9
                oper2 = (inst[0]&0x000001FC)>>2
            if fullinst:
                oper4 = (inst[1]&0x001fc000)>>14
            else:
                oper4 = None
            # seems that bit 6 of src has another meaning when this is a immediate
            # instruction (namely "32 bit")
            if oper3_t != None and fullinst and (inst[1]&0x00100000):
                # Operand 3 is immediate
                oper3_t = "i"
            #else:
            #    oper3_t = "r"
            
            # Process parameter numbers in shared memory
            if fullinst and (inst[1]&0x00200000) and not print_msize:
                if (flags&0x10):
                    oper1_t = "yo" # from offset reg
                else:
                    oper1_t = "y"
            elif flags&4:
                oper1_t = "yw"
                #if oper1 >= 0x30: # Weird mapping for 32 bit
                #    oper1 += 0x30
                #else:
                #    oper1 += 0x10

            if immediate:
                # Immediate data
                oper3 = ((inst[1]&0x0FFFFFFC)<<4) | ((inst[0]&0x003F0000)>>16)
                oper3_t = "i"
                #rv.write("[imm %08x]" % imm)
            else:
                oper3 = ((inst[0]&0x003F0000)>>16)
                #rv.write("[imm %02x]" % imm)
            if swap_oper: # to, from
                if oper1_t != None:
                    self.print_oper(rv, oper1, oper1_t)
                if oper2_t != None:
                    rv.write(", ")
                    self.print_oper(rv, oper2, oper2_t)
            else: 
                if oper2_t != None:
                    self.print_oper(rv, oper2, oper2_t)
                if oper1_t != None:
                    rv.write(", ")
                    self.print_oper(rv, oper1, oper1_t)
            if oper3_t != None:
                rv.write(", ")
                #if flipsrc2:
                #    rv.write("-")
                self.print_oper(rv, oper3, oper3_t)
            if oper4 != None and oper4_t != None:
                rv.write(", ")
                self.print_oper(rv, oper4, oper4_t)
            if is_bra:
                # Branch instruction, divide address for convenience
                # I know the address is longer, as CUDA can address up to
                # 2Mb of kernel instructions, but I have never been 
                # able to generate a kernel this big without crashing the
                # ptxas. Probably, the higher part is in inst[1].
                addr = (inst[0]&0x0FFFFE00)>>9
                if addr&3:
                    rv.write("!nonaligned!")
                rv.write("0x%08x" % (addr)) 
            elif system:
                addr = (inst[0]&0x0FFFFE00)>>9
                rv.write("0x%08x " % (addr)) 

            # Offset (integrate this into operand)
            if flags&0x10:
                offr = (flags&0x08)>>3
                rv.write(" +ofs%i" % (offr))

            # Predication (set)
            if fullinst and (inst[1]&0x0000040):
                pred = (inst[1]&0x00000030)>>4
                rv.write(", p%i" % pred)
                       

            # Bits still unknown 
            # inst[0]:  
            #   0xF0000000 op
            #   0x0FC00000 flags
            #   0x04000000 use offset register
            #   0x02000000 \- offset register 0/1
            #   0x01000000 oper3 from constant
            #   0x00800000 oper2 from parameter (offset 0x30)
            #   0x00400000 alternative instruction, ie add->sub
            #   0x003F0000 oper3 (full instruction)
            #   0x001F0000 oper3 (short instruction or immediate)
            #   0x0000FE00 oper2
            #   0x000001FC oper1
            #   0x00000002 0=normal, 1=system (flow control)
            #   0x00000001 0=32bit, 1=64 bit
            # inst[1]:  
            #   0xE0000000 subop
            #   0x08000000 signed/unsigned
            #   0x04000000 32/16
            #   0x01000000 oper3 is immediate
            #   0x00E00000 type, on load instructions
            #   0x00400000 oper3/4 from constant in segment 1
            #   0x00200000 oper2 from parameter
            #   0x001FC000 oper4 or sub-sub op
            #   0x00003000 predicate to act on
            #   0x00000400 ?? (usually set, unless predicated)
            #   0x00000200 execute on pred
            #   0x00000100 execute on !pred
            #   0x00000080 ?? (usually set, unless predicated)
            #   0x00000040 set predicate
            #   0x00000030 predicate to set
            #   0x0000003C atomic op
            #   0x00000003 marker (0=normal,1=end,2???,3=immediate)
            # to find: predication
            
            rv.write("\n")
        return rv.getvalue()

class Dummy:
    """Dummy environment that absorbs environments we are not interested in"""
    def extend(self, x):
        pass

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
                elif cmd == "consts" or cmd == "mem" or cmd == "sampler" or cmd == "const":
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
            setattr(inside[-1], valname, valvalue)
        else:
            # Bincode?
            inst = line.strip().split(" ")
            inst = [int(x,0) for x in inst]
            inside[-1].extend(inst)
            #print "inst", inst
            
    return ex
       
if __name__ == "__main__":       
    import sys
    cu = load(sys.argv[1])
    print cu.kernels[0].disassemble()

