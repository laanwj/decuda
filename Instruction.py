#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007

from cStringIO import StringIO
from Operand import *
from Formatter import *
from Constants import *
from Util import numlookup
import re

OPER_MSIZE = 1
OPER_PSIZE = 2
OPER_PSIZE3 = 4
OPER_FLOAT = 5
OPER_FLOAT32 = 5
OPER_FLOAT16 = 6
OPER_VARFLOAT = 7 # f16 or f32
OPER_FLOAT64 = 8


_msize = [
(OP_SIGN_UNSIGNED,8),  # .u8
(OP_SIGN_SIGNED,8),    # .s8
(OP_SIGN_UNSIGNED,16), # .u16
(OP_SIGN_SIGNED,16),   # .s16
(OP_SIGN_NONE,64),     # .b64
(OP_SIGN_NONE,128),    # .b128
(OP_SIGN_UNSIGNED,32), # .u32
(OP_SIGN_SIGNED,32)    # .s32
]


class DecodeError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

mask_cache = {}
def mask_to_shift(x):
    """
    Shift right the mask until the lowest bit reaches bit 0. Cache results
    in a dictionary, as the number of used masks is limited and this function 
    is used a lot.
    """
    try:
        return mask_cache[x]
    except KeyError:
        for i in xrange(0,32):
            if x&(1<<i):
                mask_cache[x] = i
                return i
        mask_cache[x] = 32
        return 32

class Instruction(object):
    # Address in cubin
    address = None
    line = None  # Line number in assembly file
    # Representation
    inst = None
    # Accounting
    visited = None
    # Decode parameters
    fullinst = None
    immediate = None
    system = None
    op = None
    # predication (in)
    pred = None
    pred_op = None  # condition code
    # Decoded
    base = None
    modifiers = None
    dst_operands = None
    src_operands = None # XX need a way to change preferred order
    warnings = None # Warnings when full decompilation fails
    # Attributes
    predicated = True # Can this instruction be predicated?
    
    @property
    def subop(self):
        if len(self.inst)==2: # has subop
            return self.bits(1,0xe0000000)
        else:
            return 0

    @property
    def subsubop(self):
        if self.fullinst: # has subsubop
            return self.bits(1,0x001fc000)
        else:
            return None
    
    def __init__(self, i=None):
        self.modifiers = []
        self.operands = []
        self.warnings = []
        self.dst_operands = []
        self.src_operands = []
        self.visited = [0x00000000, 0x00000000]
    
        if i!=None:
            self.address = i.address
            self.visited = i.visited
            self.inst = i.inst
        
    def bits(self, i, mask):
        self.visited[i] |= mask
        shift = mask_to_shift(mask)
        return (self.inst[i]&mask)>>shift
        
    def parse(self, text):
        """Parse instruction text"""
        # [@<pred>] <base>[.<mod1>[.<mod2>...] <dest>[|<dest2>], <src1>, <src2>, ...
        text = text.strip()
        
        # Predication
        if text.startswith('@'):
            # Find and remove predicate string
            pred_end = text.index(" ")
            pred = text[1:pred_end]
            text = text[pred_end+1:]

            match = re.match("(\!?)\$p([0-9]+)\.([0-9a-z]+)", pred)
            if not match:
                raise ValueError("Invalid predicate expression %s" % pred)
            (neg, pred, cc) = match.groups()
            # [!]$<pred>.<cc>
            try:
                cc = numlookup(condition_codes_rev, cc)
            except KeyError:
                raise ValueError("Invalid predicate condition code %s" % cc)
            if neg == "!":
                # Invert predicate
                cc = cc ^ 0xF
            # XXX we should be able to leave out the condition code if we just want
            # to evaluate the output of a set instruction
            self.pred_op = cc
            self.pred = int(pred)
        else:
            self.pred_op = 15 # always true
            self.pred = 0
        
        try:
            base_end = text.index(" ")
            ins = text[0:base_end]
            args = text[base_end+1:].split(",")
        except ValueError:
            ins = text
            args = []
        mods = ins.split(".")
        self.base = mods[0]
        mods = ["."+x for x in mods[1:]]
        
        # Filter out and separate types
        types = []
        typere = re.compile("\.([subf][0-9]+|label)")
        for x in mods:
            if typere.match(x):
                types.append(x)
            else:
                self.modifiers.append(x)

        # Parse operands
        try:
            dest = args[0]
            dst_args = [x.strip() for x in dest.split("|")]
        except IndexError:
            dst_args = []
        src_args = [x.strip() for x in args[1:]]
        
        i = 0
        for x in dst_args:
            op = Operand()
            if x.startswith("$p"):
                # predicates have no type specifier
                type = ""
            else:
                try:
                    type = types[i]
                except IndexError:
                    raise ValueError("Not enough operand types in instruction")
                if len(types)!=1:
                    i += 1
            op.parse(x, type)
            self.dst_operands.append(op)
        
        for x in src_args:
            op = Operand()
            if x.startswith("$p"):
                # predicates have no type specifier
                type = ""
            else:
                try:
                    type = types[i]
                except IndexError:
                    raise ValueError("Not enough operand types in instruction")
                if len(types)!=1:
                    i += 1
            op.parse(x, type)
            self.src_operands.append(op)

    def decode(self):
        #self.inst  = inst
        
        self.op    = self.bits(0,0xF0000000)
        # Large embedded operand
        if len(self.inst) == 2 and self.bits(1,3)==3:
            self.immediate = True
        else:
            self.immediate = False
        # Flow control instruction
        if self.bits(0,2):
            self.system = True
        else:
            self.system = False
        # Is a second, full word present?
        self.fullinst = (len(self.inst)==2) and not self.immediate
        # Changes some instructions
        # Seems to choose an alternative instruction set, for example,
        # add becomes sub. It is also used on mul24.lo.s32 sometimes
        # but I have no idea what its effect is there.
        # inst.alt = (inst.flags & 1)
        
        # Predication
        if self.fullinst: 
            # Predicated execution
            self.pred_op = self.bits(1,0x00000F80)
            self.pred = self.bits(1,0x00003000)
            # Predication (set)
            # This should be an operand
            if self.bits(1,0x0000040):
                self.dst_operands.append(Operand(
                  OP_TYPE_PRED, OP_SIGN_NONE, 1, OP_SOURCE_PRED_REGISTER, 
                  OP_INDIRECTION_NONE, self.bits(1,0x00000030)))
                
        else:
            self.pred_op = None # Don't get in the way
            
        # Ignore out
        #if self.fullinst:
        #    self.output_reg = self.bits(1,0x0000008)
        #else:
        #    self.output_reg = False
            
        if self.bits(0,3)==2:
            self.warnings.append("Unknown marker 0.2")
        if self.fullinst and self.bits(1,3)==2:
            #self.warnings.append("Unknown marker 1.2")
            # join point?
            self.modifiers.append(".join")
        if self.fullinst and self.bits(1,3)==1:
            self.modifiers.append(".end")
            
        if len(self.inst)==1:
            self.modifiers.append(".half")

    def default_oper_type(self, oper_type):
        # Find overall type/sign/size of instruction
        if oper_type == OPER_MSIZE:
            type = OP_TYPE_INT
            (sign, size) = _msize[self.bits(1,0x00E00000)]
        elif oper_type == OPER_PSIZE:
            type = OP_TYPE_INT
            if self.fullinst:
                if self.bits(1,0x08000000):
                    sign = OP_SIGN_SIGNED
                else:
                    sign = OP_SIGN_UNSIGNED
                if self.bits(1,0x04000000):
                    size = 32
                else:
                    size = 16
            else:
                sign = OP_SIGN_NONE
                if self.bits(0,0x00008000):
                    size = 32
                else:
                    size = 16
        elif oper_type == OPER_PSIZE3: # psize, width only
            type = OP_TYPE_INT
            sign = OP_SIGN_NONE
            if self.fullinst:
                if self.bits(1,0x04000000):
                    size = 32
                else:
                    size = 16
            else:
                if self.bits(0,0x00008000):
                    size = 32
                else:
                    size = 16
        elif oper_type == OPER_FLOAT:
            sign = OP_SIGN_NONE
            size = 32
            type = OP_TYPE_FLOAT
        elif oper_type == OPER_FLOAT16:
            sign = OP_SIGN_NONE
            size = 16
            type = OP_TYPE_FLOAT
        elif oper_type == OPER_FLOAT64:
        	sign = OP_SIGN_NONE
        	size = 64
        	type = OP_TYPE_FLOAT
        elif oper_type == OPER_VARFLOAT:
            sign = OP_SIGN_NONE
            if self.bits(1, 0x04000000):
                size = 32
            elif self.bits(1, 0x00400000):
                size = 64
            else:
                size = 16
            type = OP_TYPE_FLOAT
        return (sign, size, type)

    def offset_reg(self):
        """ofs1-4 offset register used by this instruction?""" 
        t_offset = self.bits(0,0x0C000000)
        if self.fullinst:
            t_offset |= self.bits(1,0x00000004)<<2
        return t_offset

    def decode_operand(self, oper_type, operid, msize_bits=True, imm=False, indir=None, reg=False, regscale=True, doffset=False, spred=False, flip=False, tex=False, opt_imm=False):
        """What operands do we want, and in which order
        operid 1 is the first source operand in word 0
               2 is the destination operand (usually)
               3 immediate(spans both words) or non immediate (upper part of first word)
               4 subsub operation or fourth operand
               5 1 and 3 merged
        """
        (sign, size, type) = oper_type
        
        source = OP_SOURCE_REGISTER
        indirection = OP_INDIRECTION_NONE
        offset = OP_OFFSET_NONE
        multiplier = False
        offset_inc = False

        # Offset (integrate this into operand)
        #t_offset = (self.flags&0x30)>>4
        t_offset = self.offset_reg()
        
        if operid == 1: # first source operand, in main word
            if self.fullinst: # Depending on this, source is 7 or 6 bits
                # seems that bit 6 of oper1 has another meaning when this is a immediate
                # instruction (namely "32 bit")
                value = self.bits(0,0x0000FE00)
            else:
                value = self.bits(0,0x00007E00)

            # Process parameter numbers in shared memory
            if self.fullinst and self.bits(1,0x00200000) and msize_bits: # MSIZE bits collide with these
                indirection = OP_INDIRECTION_SHARED
                source = OP_SOURCE_IMMEDIATE
                
                t = (value&0x60)>>5
                value &= 0x1F
                (nsign, nsize, _) = sharedmem_types[t]
                if nsize != size: # XXX propagate sign
                    self.warnings.append("shared memory operand type mismatch")
            if not self.fullinst and self.bits(0,0x01000000):
                indirection = OP_INDIRECTION_SHARED
                source = OP_SOURCE_IMMEDIATE
                
                t = (value&0x30)>>4
                value &= 0xF
                (nsign, nsize, _) = sharedmem_types[t]
                if nsize != size: # XXX propagate sign
                    self.warnings.append("shared memory operand type mismatch")
        elif operid == 2: # destination operand, in main word
            value = self.bits(0,0x000001FC)
            if self.fullinst and self.bits(1,0x0000008):
                # to output register
                source = OP_SOURCE_OUTPUT_REGISTER
        elif operid == 3: # immediate(spans both words) or non immediate (upper part of first word)
            # Extract and print operands
            cflag = False
            if self.bits(0,0x00800000):
                # operand 3 comes from constant (in segment 0)
                indirection = OP_INDIRECTION_CONST0
                source = OP_SOURCE_IMMEDIATE

                #if self.fullinst and self.bits(1,0x00400000) and msize_bits: # MSIZE bits collide with these
                #    # operand 3 or 4 comes from constant in segment 1
                #    indirection = OP_INDIRECTION_CONST1
                #    source = OP_SOURCE_IMMEDIATE
                #    #self.warnings.append("constbit")
                if self.fullinst and msize_bits:
                    indirection += self.bits(1,0x00C00000)
                if not self.fullinst:
                    indirection += self.bits(0,0x00200000)
                    cflag = True
            if self.immediate:
                # Immediate data
                value = (self.bits(1,0x0FFFFFFC)<<6) | self.bits(0,0x003F0000)
                source = OP_SOURCE_IMMEDIATE
            else:
                if cflag:
                    # half instruction, upper bit is segment
                    value = self.bits(0,0x001F0000)
                else:
                    value = self.bits(0,0x007F0000)
            
            if opt_imm and self.fullinst and self.bits(1,0x00100000):
                # Operand 3 is immediate
                source = OP_SOURCE_IMMEDIATE

        elif operid == 4:  # sub operation or fourth operand
            if not self.fullinst:
                raise DecodeError("No operand 4 in this instruction")
            #if (inst[0]&0x00800000):
            #    # operand 3 comes from constant (in segment 0)
            #    indirection = OP_INDIRECTION_CONST0
            #    source = OP_SOURCE_IMMEDIATE
            if self.fullinst and self.bits(0,0x01000000):
                # operand 4 comes from constant (in seg 0)
                indirection = OP_INDIRECTION_CONST0
                source = OP_SOURCE_IMMEDIATE

                #if self.fullinst and self.bits(1,0x00400000) and msize_bits: # MSIZE bits collide with these
                #    # operand 3 or 4 comes from constant in segment 1
                #    indirection = OP_INDIRECTION_CONST1
                #    source = OP_SOURCE_IMMEDIATE
                #    #self.warnings.append("constbit")
                if self.fullinst and msize_bits:
                    indirection += self.bits(1,0x03C00000)
        
            value = self.bits(1,0x001fc000)
            
        elif operid == 5: # use as much of word 0 as possible by merging 1 and 3
            source = OP_SOURCE_IMMEDIATE
            value = self.bits(0,0x003FFE00)
        elif operid == 6:
            source = OP_SOURCE_OFFSET_REGISTER
            value = t_offset
            
        # Sometimes we need to force immediate operand
        if imm:
            source = OP_SOURCE_IMMEDIATE        
        # Force register
        if reg:
            source = OP_SOURCE_REGISTER
            indirection = OP_INDIRECTION_NONE
        # Set some indirection based on the instruction
        if indir != None:
            indirection = indir
        # Offset (integrate this into operand)
        if indirection in [OP_INDIRECTION_LOCAL,OP_INDIRECTION_GLOBAL,OP_INDIRECTION_CONST0,OP_INDIRECTION_CONST1,OP_INDIRECTION_SHARED] and t_offset:
            # ld.offset0.b32 $r01 -> 0x10
            # ld.offset1.shl.b32 $r01 -> 0x10
            # ld.offset1.shl.b32 $r02 -> 0x20
            source = OP_SOURCE_IMMEDIATE
            offset = t_offset
            if self.bits(0,0x02000000):
                offset_inc = True
        # Scale registers if half
        if regscale and source==OP_SOURCE_REGISTER and size in [8,16] and indirection == OP_INDIRECTION_NONE:
            source = OP_SOURCE_HALF_REGISTER
        # Dest offset register?
        if doffset:
            source = OP_SOURCE_OFFSET_REGISTER
        # Pred register
        if spred:
            source = OP_SOURCE_PRED_REGISTER
        # Texture
        if tex:
            source = OP_SOURCE_TEXTURE
        # Address multiplier
        if source == OP_SOURCE_IMMEDIATE and (indirection == OP_INDIRECTION_SHARED or (indirection >= OP_INDIRECTION_CONST0 and indirection <= OP_INDIRECTION_CONST15)):
            multiplier = None # ??
            if size == 8:
                multiplier = 1
            elif size == 16:
                multiplier = 2
            elif size == 32:
                multiplier = 4
            if multiplier != None:
                value *= multiplier
            else:
                self.warnings.append("Invalid multiplier")
        
        return Operand(type, sign, size, source, indirection, value, offset, flip, offset_inc)
        
    def unknown_subsubop(self):
        #self.modifiers.append(".%02x?" % self.subsubop)
        if self.subsubop != None:
            self.warnings.append("Unknown subsubop %02x" % self.subsubop)
        else:
            self.warnings.append("Did not expect subsubop to be None")
        
    def __repr__(self):
        rv = StringIO()
        self.dump(rv)
        return rv.getvalue()
        
    def dump(self, rv, fmt=Formatter()):
        # Predication
        # Condition code
        # What do these mean?
        # do we have a zero bit, sign bit?
        # self.pred_op&3 seems straightforward enough
        # but what is self.pred_op>>2  ?
        if not self.predicated or self.pred_op == 15 or self.pred_op == None:
            pass # No self.pred
        else:
            fmt.pred(rv, "@$p%i.%s" % (self.pred, condition_codes[self.pred_op]))
        #elif self.pred_op == 2:  # 0010
        #    # Execute on false, not on true
        #   fmt.pred(rv, "@!$p%i" % self.pred)
        #elif self.pred_op == 5:  # 0101
        #    # Execute on true, not on false
        #    fmt.pred(rv, "@$p%i" % self.pred)
        #elif (self.pred_op&3)==2: # xx10  -- seems to be @!$p%i
        #    fmt.pred(rv, "@!$p%i" % (self.pred))
        #    self.warnings.append("cc is %s" % condition_codes[self.pred_op])
        #elif (self.pred_op&3)==1: # xx01  -- seems to be @p%i
        #    fmt.pred(rv, "@$p%i" % (self.pred))
        #    self.warnings.append("cc is %s" % condition_codes[self.pred_op])
        #elif logic_ops.has_key(self.pred_op):
        #    # unsigned version
        #    fmt.pred(rv, "@$p%i%s.u" % (self.pred,logic_ops[self.pred_op]))
        #elif (self.pred_op>=8 and self.pred_op<16) and logic_ops.has_key(self.pred_op-8):
        #    # signed version
        #    fmt.pred(rv, "@$p%i%s.s" % (self.pred,logic_ops[self.pred_op-8]))
        #else:
        #    fmt.pred(rv, "@$p%i.%i" % (self.pred, self.pred_op))
        #    self.warnings.append("cc is %s" % condition_codes[self.pred_op])
        
        # Base
        if self.base:
            fmt.base(rv, self.base)
        elif self.system:
            fmt.base(rv, "sop.%01x" % (self.op))
        else:
            fmt.base(rv, "op.%01x%01x" % (self.op,self.subop))
        # Add instruction modifiers
        for m in self.modifiers:
            fmt.modifier(rv, m)
        # Operand types
        # collapse if all are the same
        
        #optypes.extend([x.typestr() for x in self.dst_operands])
        
        # Promote sign of source operands
        srco = self.dst_operands + self.src_operands
        srco = [x.clone() for x in srco]
        sign = OP_SIGN_NONE
        for o in srco:
            if o.sign != OP_SIGN_NONE:
                sign = o.sign
        for o in srco:
            if o.sign == OP_SIGN_NONE:
                o.sign = sign

        # Add to operand list
        optypes = []
        optypes.extend([x.typestr() for x in srco])
        optypes = [x for x in optypes if x!=""] # Filter empty types (predicates)
        oset = set(optypes)

        if len(oset) == 1:
            # There is only one type
            fmt.types(rv, optypes[0])
        else:
            # Show all types
            fmt.types(rv, "".join(optypes))
        # Destination operands
        dst_operands = self.dst_operands[:]
        #if self.ignore_result:
        #    # When ignoring result, only pred register output
        #    dst_operands = [x for x in dst_operands if x.source==OP_SOURCE_PRED_REGISTER]
        # output register 0x7f = bit bucket
        # dst_operands = [x for x in dst_operands if not (x.source==OP_SOURCE_OUTPUT_REGISTER and x.value==0x7f)]
        if len(dst_operands):
            operands_str = [x.__repr__() for x in dst_operands]
            fmt.dest_operands(rv, "|".join(operands_str))
        # Source operands
        if len(self.src_operands):
            pre = ""
            if len(self.dst_operands):
                pre = ", "
            operands_str = [x.__repr__() for x in self.src_operands]
            fmt.src_operands(rv, pre+(", ".join(operands_str)))
        # Disassembler warnings
        if self.inst != None and self.visited != None:
            unk0 = self.inst[0] & ~(self.visited[0])
            if len(self.inst)==2:
                unk1 = self.inst[1] & ~(self.visited[1])
            else:
                unk1 = 0
            if unk0:
                fmt.warning(rv, "unk0 %08x" % (unk0))
            if unk1:
                fmt.warning(rv, "unk1 %08x" % (unk1))

        for warn in self.warnings:
            fmt.warning(rv, warn)

    def assemble(self):
        from Assembler import assemble
        self.inst = assemble(self)
        
