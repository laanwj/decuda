#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan, 2007
from Operand import *
from Instruction import *
from CubinFile import *
from AsmConstants import *
from AsmRules import rules # import just the rules table
from Exceptions import *
from Util import wraplist

# Parse PTX like syntax, assemble bit format, but will not be a replacement for 
# PTX as we don't optimize. Look at llvm for that.
# operand types and sizes allowed for moves from/to global and local
_msize_rev = {
(OP_SIGN_NONE,8):0,      # .b8
(OP_SIGN_UNSIGNED,8):0,  # .u8
(OP_SIGN_SIGNED,8):1,    # .s8
(OP_SIGN_NONE,16):2,     # .b16
(OP_SIGN_UNSIGNED,16):2, # .u16
(OP_SIGN_SIGNED,16):3,   # .s16
(OP_SIGN_NONE,64):4,     # .b64
(OP_SIGN_NONE,128):5,    # .b128
(OP_SIGN_NONE,32):6,     # .b32
(OP_SIGN_UNSIGNED,32):6, # .u32
(OP_SIGN_SIGNED,32):7    # .s32
}
_cvti_types_rev = {
(OP_SIGN_UNSIGNED,16,OP_TYPE_INT):0x0,  # .u16
(OP_SIGN_UNSIGNED,32,OP_TYPE_INT):0x1,  # .u32
(OP_SIGN_UNSIGNED,8,OP_TYPE_INT):0x2,   # .u8
(OP_SIGN_SIGNED,16,OP_TYPE_INT):0x4,    # .s16
(OP_SIGN_SIGNED,32,OP_TYPE_INT):0x5,    # .s32
(OP_SIGN_SIGNED,8,OP_TYPE_INT):0x6      # .s8
}

def match_type(a,b):
    """Match operand type against operand type rule"""
    b = (b.type, b.sign, b.size, b.source, b.indirection)
    #print a,b
    for aa,bb in zip(a,b):
        if not bb in wraplist(aa):
            return False
    return True

def align_shift(size, val):
    """Alignment shift for data of a certain size"""
    if size == 32:
        if val&3:
            raise ValueError("32 bit operand not aligned to 4 bytes")
        return val>>2
    elif size == 16:
        if val&1:
            raise ValueError("16 bit operand not aligned to 2 bytes")
        return val>>1
    elif size == 8:
        return val>>0
    else:
        raise ValueError("Invalid data size %i" % size)

def assemble(i):
    error = "Unknown instruction"
    # Preprocess the instruction here, this will save us quite some hacks 
    # and grief later on.
    if i.base == "d":
        # special
        return [x.value for x in (i.dst_operands + i.src_operands)]
    operands = {}
    # enumerate dest operands
    p = DST1
    for operand in i.dst_operands:
        if operand.type != OP_TYPE_PRED:
            operands[p] = operand
            p += 1
        else:
            # output predicates are handled specially
            operands[PRED_OUT] = operand
    # enumerate source operands
    p = SRC1
    for operand in i.src_operands:
        operands[p] = operand
        p += 1
    # operand set, minus predication
    operands_set = set(operands.keys())
    operands_set.discard(PRED_OUT)
    # XXX handle required operand properties: flipped, 
    # predicate in/output. Now they just get ignored for unsupporting
    # instructions.
    
    for rule in rules:
        # Match rule 
        if rule[0] != i.base:
            continue
        #print rule[0]
        try:
            mods = rule[2][:]

            # Predicate destination arguments can be filtered out
            # OUT_PRED and not DSTx
            #nonpred = [o for o in i.dst_operands if o.type != OP_TYPE_PRED]
            # Evaluate arguments, make sure they are all 'touched'
            iset = set()
            for arg, type, bits in rule[3]:
                #print arg, type, bits
                try:
                    obj = operands[arg]
                except LookupError:
                    raise ValueError("Missing required argument")
                if match_type(type, obj):
                    mods += bits
                    iset.add(arg)
            # iset contains all matched arguments (excluding PRED_IN, PRED_OUT)
            # operands_set contains all arguments
            if iset != operands_set:
                # Not all arguments matched
                # we should do better error reporting and report which arguments don't match
                raise ValueError("Invalid argument types")
            # Now that mods in complete, check if required attributes are 'touched':
            #   PRED_IN, PRED_OUT, flip, invert, offset, offset_inc
            
            # Instruction matched
            # synthesize bit field, process modifiers
            inst = [0x00000000] * rule[1]
            used = [0x00000000] * rule[1]
            r_modifiers = set(i.modifiers) # remaining modifiers
            for b in mods:
                #print b
                bf, src, sub = b[0], b[1], b[2]
                bits = None
                if len(b)==4:
                    # bit filter specified
                    bits = b[3]
                if src == IMM:
                    value = sub
                elif (src >= DST1 and src <= SRC_LAST) or src == PRED_OUT:
                    # destination or source attribute
                    try:
                        obj = operands[src]
                    except LookupError:
                        #if sub == PRESENT:
                        value = 0
                        #else:
                        #    raise ValueError("Missing argument %i" % src)
                    else:
                        if sub == PRESENT:
                            value = 1
                        elif sub == VALUE:
                            value = obj.value
                        elif sub == VALUE_ALIGN:
                            value = align_shift(obj.size, obj.value)
                        elif sub == SHTYPE:
                            if obj.size == 8:
                                value = 0
                            elif obj.size == 16:
                                if obj.sign == OP_SIGN_SIGNED:
                                    value = 2
                                else:
                                    value = 1
                            elif obj.size == 32:
                                value = 3
                            else:
                                raise ValueError("Invalid shared memory operand type")
                        elif sub == OFFSET:
                            if obj.offset != None:
                                value = obj.offset
                            else:
                                value = 0 # no offset = offset reg 0
                        elif sub == OFFSET_INC:
                            value = obj.offset_inc
                        elif sub == FLIP:
                            value = obj.flip
                        elif sub == INVERT:
                            value = obj.invert
                        elif sub == CONSTSEG:
                            value = obj.indirection - OP_INDIRECTION_CONST0
                        elif sub == IS_SIGNED:
                            value = (obj.sign == OP_SIGN_SIGNED)
                        elif sub == IS_32BIT:
                            if obj.indirection == OP_INDIRECTION_NONE:
                                if obj.size == 16 and obj.source in [OP_SOURCE_REGISTER, OP_SOURCE_OUTPUT_REGISTER]:
                                    raise ValueError("Type conflict -- expected half register")
                                if obj.size == 32 and obj.source in [OP_SOURCE_HALF_REGISTER, OP_SOURCE_HALF_OUTPUT_REGISTER]:
                                    raise ValueError("Type conflict -- expected full register")
                            value = (obj.size == 32)
                        elif sub == GET_MSIZE:
                            value = _msize_rev[(obj.sign,obj.size)]
                        elif sub == IS_OUTREG: # operand is normal or output register
                            if (obj.indirection != OP_INDIRECTION_NONE or 
                               not obj.source in [OP_SOURCE_HALF_OUTPUT_REGISTER, OP_SOURCE_OUTPUT_REGISTER, OP_SOURCE_HALF_REGISTER, OP_SOURCE_REGISTER]):
                                raise ValueError("Output register operand must be register")
                            value = obj.source in [OP_SOURCE_HALF_OUTPUT_REGISTER, OP_SOURCE_OUTPUT_REGISTER]
                        elif sub == CVTI_TYPE: # conversion type
                            value = _cvti_types_rev[(obj.sign, obj.size, obj.type)]
                        else:
                            raise ValueError("Invalid sub value %i" % sub)
                elif src == MODIFIER:
                    # check for presence of modifier
                    if sub in i.modifiers:
                        value = bits
                        try:
                            r_modifiers.remove(sub)
                        except KeyError:
                            pass # multiply occurences the same modifier are allowed
                    else:
                        if bf == BF_ALWAYS:
                            # Required value is not present
                            raise ValueError("Required modifier not present")
                        continue # modifier not there, just continue with next rule
                elif src == PRED_IN:
                    if sub == CC:
                        value = i.pred_op
                    elif sub == VALUE:
                        value = i.pred
                    else:
                        raise ValueError("Invalid sub value for PRED_IN %i" % sub)
                else:
                    raise ValueError("Invalid source value")
                # in case of BF_ALWAYS, don't set anything
                if bf != BF_ALWAYS:
                    #print "%i %08x %08x" % (bf[0], bf[1], value)
                    if bits != None and src != MODIFIER:
                        # select part of the bits
                        shift = mask_to_shift(bits)
                        value = (value & bits)>>shift
                    shift = mask_to_shift(bf[1])
                    value = value << shift
                    if (value & bf[1])!=value:
                        raise ValueError("Operand does not fit")
                    if used[bf[0]] & bf[1]: # bits were already set by another modifier?
                        # if we're trying to set it to something different, this is an collision
                        if (inst[bf[0]] & bf[1]) != value: 
                            raise ValueError("Bit collision")

                    inst[bf[0]] |= value
                    used[bf[0]] |= bf[1]
                
            if len(r_modifiers):
                raise ValueError("Unknown or unsupported modifiers "+("".join(r_modifiers)))
            return inst
        except ValueError,e:
            error = e.message
            # store error, but try next rule
            # XXX cope with errors in a smarter way, as we don't always display the most
            # interesting error now
    raise CompilationError(i.line, error) # re-raise error if we didn't find any matching rule


class Assembler(object):
    comment = ["#", "//"]  # comment start
    def __init__(self):
        self.output = CubinFile()
    def assemble(self, i):
        # Parsing phase
        # XXX better error reporting, it should at least print the line
        kernel = None
        state = 0 # closed
        line = 0 # line number
        while True:
            line += 1
            text = i.readline()
            if not text:
                break
            text = text.rstrip("\r\n")
            # strip comments
            for c in self.comment:
                try:
                    text = text[:text.index(c)]
                except ValueError:
                    pass
            # strip trailing or initial whitespace
            text = text.lstrip("\t ")
            text = text.rstrip("\t ")
            # skip empty lines or lines containing only comments
            if len(text)==0:
                continue
            if text == "{":
                # open block
                if state != 0 or kernel == None:
                    raise CompilationError(line, "Block open in wrong context")
                state = 1 # block opened
            elif text == "}":
                # close block
                if state != 1 or kernel == None:
                    raise CompilationError(line, "Block close in wrong context")
                #print kernel.instructions
                self.output.kernels.append(kernel)
                state = 0 # closed
                kernel = None
            elif text.startswith("."):
                # meta instruction
                text = text[1:]
                inst = text.split(" ")
                if inst[0] == "entry":
                    kernel = Kernel()
                    kernel.name = inst[1]
                    kernel.lmem = 0
                    kernel.smem = 0
                    kernel.reg = 0
                    kernel.bar = 0
                    kernel.instructions = []
                elif inst[0] in ["lmem", "smem", "reg", "bar"]:
                    if kernel == None:
                        raise CompilationError(line, "Kernel attribute outside kernel definition")
                    setattr(kernel, inst[0], int(inst[1]))
                elif inst[0] == "constseg":
                    # start of constant segment
                    # N:offset name
                    (N, offset) = inst[1].split(":")
                    try:
                        name = inst[2]
                    except LookupError:
                        name = None
                    print N, offset, name
                else:
                    raise CompilationError(line, "Invalid meta-instruction %s" % inst[0])
            else: 
                # check for label
                try:
                    lend = text.index(":")
                except ValueError:
                    pass
                else:
                    # label
                    label = text[0:lend]
                    text = text[lend+1:]
                    text = text.lstrip("\t ")
                    ptr = len(kernel.instructions)
                    kernel.instructions.append(Label(label))
                    
                if len(text) != 0:
                    # normal instruction
                    if kernel == None:
                        raise CompilationError(line, "Instruction outside kernel definition")
                    inst = Instruction()
                    inst.visited = None
                    inst.line = line
                    try:
                        inst.parse(text)
                    except ValueError,e:
                        raise CompilationError(line, e.message)
                    kernel.instructions.append(inst)
                    #inst.inst = assemble(inst)
                    #kernel.bincode.extend(inst.inst)
        
        # Assembly phase
        for k in self.output.kernels:
            k.assemble()
        #self.output.kernels.append(kernel)

def test_asm(x):
    i = Instruction()
    i.visited = None
    i.parse(x)
    
    # Print instruction, to test parsing/dumping
    print "in                %s" % (i)
    # Assemble
    i.inst = assemble(i)
    #print "%08x %08x" %(i.inst[0], i.inst[1])
    # Then disassemble again, and see if we get similar result
    d = Disassembler()
    i2 = d.decode(0, i.inst)

    #print i2
    
    print "%08x %08x %s" % (i.inst[0], i.inst[1], i2)
    print

def main():
    test_asm("set.eq.s16 $p0|$r2.lo, $r1.lo, c0[$ofs1+$r1]")
    test_asm("shl.s16 $r2.lo, $r1.lo, 0x12")
    test_asm("shr.s32 $r2, $r1, 0x12")
    test_asm("nop")
    test_asm("d.u32 0x211ce801, 0x00000007")
    test_asm("mov.b32 $r1, $r2")
    test_asm("mov.b32 $o1, $r2")
    test_asm("mov.b32 $r1, s[$ofs1+$r1]")
    test_asm("mov.u32 $r1, c0[$ofs1+$r1]")
    test_asm("mov.b32 s[0x10], $r1")
    test_asm("mov.b16 s[0x10], $r1.lo")
    test_asm("mov.b8.b16 s[0x10], $r1.lo")
    test_asm("mov.b32 $r1, $p0")
    test_asm("mov.b32 $r1, $ofs1")
    test_asm("mov.b32 $p1, $r1")
    test_asm("mov.b32 $r1, %physid")
    test_asm("add.b32 $ofs2, $ofs1, 0x12")
    test_asm("mov.b32 g[$r0], $r0")
    test_asm("mov.b32 l[$r0], $r0")
    test_asm("mov.b32 $r0, l[$r1]")
    test_asm("mov.b64 $r0, g[$r1]")
    test_asm("mov.b32 $r1, 0x123456")
    test_asm("mov.b32 $r0, s[0x0014]")
    test_asm("cvt.u32.u8 $r0, $r1.lo") 
    test_asm("cvt.u32.u16 $r0, s[0x0006]")
    test_asm("cvt.rp.f32.s16 $r0, $r1.lo")
    test_asm("cvt.rpi.s32.f32 $r0, $r1")
    test_asm("cvt.rpi.s16.f16 $r0.lo, $r1.lo")
    test_asm("cvt.sat.f16.f32 $r0.lo, $r1")
    test_asm("cvt.sat.rni.f32.f32 $r0, $r1")
    test_asm("xor.b32 $r0, $r1, $r2")
    test_asm("not.b32 $r0, $r1")
    test_asm("xor.b16 $r0.lo, $r1.lo, $r2.lo")
    test_asm("not.b16 $r0.lo, $r1.lo")
    test_asm("neg.f32 $r1, $r0")
    test_asm("subr.rp.f32 $r1, $r3, -$r2")
    test_asm("set.ne.s32.f32.f32 $r1, $r3, $r2")
    test_asm("min.f32 $r3, $r2, $r1")
    test_asm("max.f32 $r3, $r3, $r4")
    test_asm("pre.sin.f32 $r3, $r4")
    test_asm("mul.rp.f32 $r2, $r3, $r4")    
    test_asm("cvt.scale.f32.u32 $r2, $r3")
    test_asm("mul24.hi.s32 $r2, $r3, $r4")
    test_asm("mul24.lo.u32.u16.u16 $r2, $r3.lo, $r4.lo")
    test_asm("mad24.lo.u32 $r2, $r3, $r4, $r5")
    test_asm("mad24.hi.u32 $r2, $r3, $r4, $r5")
    test_asm("mad24.lo.sat.s32 $r2, $r3, $r4, $r5")
    test_asm("mad24.lo.u32.u16.u16.u16 $r2, $r3.lo, $r4.lo, $r5.lo")
    test_asm("mad.rp.f32 $r2, $r3, $r4, $r5")
    test_asm("mad.rn.f32 $r2, s[0x0004], c0[0x14], $r5")
    test_asm("return")
    test_asm("bar.sync.u32 0x0")
    
if __name__ == "__main__":
    main()

