#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan, 2007

from Operand import *
from Instruction import *
from FlowControl import *
from Constants import *
from Util import lookup

class t01(Instruction):    
    """
    Move a predicate to a register.
    """
    def decode(self):
        super(t01,self).decode()
        
        self.base = "mov"
        
        dtype = self.default_oper_type(OPER_PSIZE)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        stype = (OP_SIGN_NONE, 1, OP_TYPE_PRED)
        self.src_operands.append(self.decode_operand(stype, 1, spred=True))

class t02(Instruction):    
    """
    Move an offset register to a register.
    """
    def decode(self):
        super(t02,self).decode()
        
        self.base = "mov"
        
        type = (OP_SIGN_NONE, 32, OP_TYPE_INT)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 6))


class t05(Instruction):    
    """
    Move an register to a predicate. The destination predicate is marked
    as destination predicate of the instruction, the source register is arg1.
    """
    def decode(self):
        super(t05,self).decode()
        
        self.base = "mov"

        type = self.default_oper_type(OPER_PSIZE)
        #self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))

class ldgpu(Instruction):    
    """Load an internal value"""
    def decode(self):
        super(ldgpu,self).decode()
        
        self.base = "mov"
        
        type = (OP_SIGN_NONE,32,OP_TYPE_INT)
        self.src_operands.append(self.decode_operand(type, 4, indir=OP_INDIRECTION_INTERNAL, imm=True))
        self.dst_operands.append(self.decode_operand(type, 2))
        

class add(Instruction):
    """Add two integer operands"""
    def decode(self):
        super(add,self).decode()
        self.base = "add"
        if self.bits(0,0x00400000):
            self.base = "sub"
        else:
            self.base = "add"
        if self.fullinst and self.bits(1,0x08000000):
            # .s32 also signifies saturation
            self.modifiers.append(".sat")
            
        type = self.default_oper_type(OPER_PSIZE)
        
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        if self.fullinst:
            # oper 2,1,4
            self.src_operands.append(self.decode_operand(type, 4))
        else:
            # oper 2,1,3
            self.src_operands.append(self.decode_operand(type, 3))

class ldofs1(Instruction):
    """Set offset pointer1 to (a<<b)"""
    def decode(self):
        super(ldofs1, self).decode()
        self.base = "movsh"
        
        type = (OP_SIGN_NONE,32,OP_TYPE_INT)
        
        self.dst_operands.append(self.decode_operand(type, 2, doffset=True))
        self.src_operands.append(self.decode_operand(type, 1))
        self.src_operands.append(self.decode_operand(type, 3, imm=True))

class stsha(Instruction):
    """Store a value to shared memory"""
    def decode(self):
        super(stsha, self).decode()
        self.base = "mov"
        
        type = self.bits(1,0x00600000) # dst width
        if type == 0:
            size = 16
        elif type == 1: 
            size = 32
        elif type == 2:
            size = 8
        else: # ??
            self.modifiers.append(".?%i?" % type)
            size = 0

        atom = self.bits(1,0x00800000)
        if atom:
            self.modifiers.append(".atom")

            
        #flag =  self.bits(0,0x08000000) 
        #if flag:
        #    # changes something to arguments, but what?
        #    self.warnings.append("flag 0x08000000?")
        stype = self.default_oper_type(OPER_PSIZE3)
        dtype = (OP_SIGN_NONE,size,OP_TYPE_INT)
        
        #self.dst_operands.append(self.decode_operand(dst_type, 5, indir=OP_INDIRECTION_SHARED))
        self.dst_operands.append(self.decode_operand(dtype, 5, indir=OP_INDIRECTION_SHARED, imm=True, msize_bits=False))
        self.src_operands.append(self.decode_operand(stype, 4, reg=True, msize_bits=False))
        #self.src_operands.append(self.decode_operand(dst_type, 2, reg=True))
        
        # if this instruction has an immediate addr instead of a register
        # 000308: 08021001 e4240780 st.shared.b32 s[0x0008], $r10
        # 000020: 06000201 e0400780 st.shared.b8 s[ofs1+$r01], $r00.b0
        # this immediate addr should span the entire of oper 5
        # or is it because 0x08000000 is set here?
        # seems 0x08000000 means add some offset (which?)
        # st.shared.s32   [$r27],$r40;
        # it's an register after all
        

class ld(Instruction):
    """Load data between registers, constants, ..."""
    def decode(self):
        super(ld,self).decode()
        
        if self.subsubop == 0xF or self.subsubop == None:
            self.base = "mov"
        else:
            self.unknown_subsubop()
        
        type = self.default_oper_type(OPER_PSIZE3)
        self.dst_operands.append(self.decode_operand(type, 2))

        if self.immediate:
            self.src_operands.append(self.decode_operand(type, 3))
        else:
            self.src_operands.append(self.decode_operand(type, 1))
        
class ldconst(Instruction):
    """Load data between registers, constants, ..."""
    def decode(self):
        super(ldconst,self).decode()
        
        self.base = "mov"
        
        if self.subsubop <= 0x3:
            stype = sharedmem_types[self.subsubop]
        else:
            stype = sharedmem_types[3] # ???
            self.unknown_subsubop()
        # subsubop is 0x1, 0x2 or 0x3 .. why?
        #if self.subsubop != 0x2 and self.subsubop != 0x3:
        #    self.unknown_subsubop()
        #self.warnings.append("subsubop %i" % self.subsubop)
        
        # use bit inst[1]&0x03C00000 to determine segment to load from
        indir = OP_INDIRECTION_CONST0 + self.bits(1,0x03C00000)
        
        dtype = self.default_oper_type(OPER_PSIZE3)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 5, indir=indir))

class ldshar(Instruction):
    """Load data between registers, constants, ..."""
    def decode(self):
        super(ldshar,self).decode()
        
        self.base = "mov"
        
        atom = self.bits(1,0x00800000)
        if atom:
            self.modifiers.append(".atom")

        if self.subsubop <= 0x3:
            stype = sharedmem_types[self.subsubop]
        else:
            stype = sharedmem_types[3] # ???
            self.unknown_subsubop()

        indir = OP_INDIRECTION_SHARED
        
        dtype = self.default_oper_type(OPER_PSIZE3)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 5, indir=indir))


class neg(Instruction):
    """Negate, this is more like a subtract with reversed parameters"""
    def decode(self):
        super(neg,self).decode()
        if self.bits(0,0x00400000):
            self.base = "addc"  # add with carry
        else:
            self.base = "subr"  # subtract with reversed parameters
        
        type = self.default_oper_type(OPER_PSIZE)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        if self.fullinst:
            self.src_operands.append(self.decode_operand(type, 4))
        else:
            self.src_operands.append(self.decode_operand(type, 3))

class m3x(Instruction):
    """Handler for set,max,min,shl,shr"""
    def decode(self):
        super(m3x,self).decode()
        names = {3:"set",4:"max",5:"min",6:"shl",7:"shr"}
        self.base = names[self.subop]
        if self.subop == 3:
            self.modifiers.append(lookup(logic_ops, self.subsubop))
        # else assume subsubop is 0           

        type = self.default_oper_type(OPER_PSIZE)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        self.src_operands.append(self.decode_operand(type, 3, opt_imm=True))

class mul24(Instruction):
    """24 bit precision multiplication instruction"""
    def decode(self):
        super(mul24,self).decode()
        
        self.base = "mul24"
        if self.fullinst:
            if self.subsubop & 4:
                ssize = 32
            else:
                ssize = 16
            if self.subsubop & 2:
                sign = OP_SIGN_SIGNED
            else:
                sign = OP_SIGN_UNSIGNED
            if self.subsubop & 1:
                self.modifiers.append(".hi")
            else:
                self.modifiers.append(".lo")

            if self.subsubop & (~7):
                self.unknown_subsubop()
            
            # bit should not be set for long form
            if self.bits(0,0x00400000):
                self.warnings.append("invalid combination, please report")
        else:
            if self.bits(0,0x00008000):
                sign = OP_SIGN_SIGNED
            else:
                sign = OP_SIGN_UNSIGNED
            self.modifiers.append(".lo")
        
            # bit should always be set for short form
            if self.bits(0,0x00400000):
                ssize = 32
            else:
                ssize = 16
                
        stype = (sign, ssize, OP_TYPE_INT)
        dtype = (sign, 32, OP_TYPE_INT)
                
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))
        self.src_operands.append(self.decode_operand(stype, 3))
        # opt_imm=True?

class sad(Instruction):
    """Sum of Absolute Differences"""
    def decode(self):
        super(sad,self).decode()
        
        self.base = "sad"

        type = self.default_oper_type(OPER_PSIZE)        
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        self.src_operands.append(self.decode_operand(type, 3))
        # Constant bits apply to operand 3, not 4
        self.src_operands.append(self.decode_operand(type, 4, reg=True))

# merge with mad24?
class mad16(Instruction):
    """16*16+32 bit precision multiplication+addition instruction"""
    def decode(self):
        super(mad16,self).decode()
        
        carry = self.bits(1,0x0c000000)
        if carry == 0:
            self.base = "mad24"
        elif carry == 3:
            self.base = "mad24c1" # carry from p1
        else:
            self.base = "mad24c??" # carry from p0? p2?

        dtype = (OP_SIGN_UNSIGNED,32,OP_TYPE_INT)
        
        if self.subop == 0:
            stype = (OP_SIGN_UNSIGNED,16,OP_TYPE_INT)
        elif self.subop == 1:
            stype = (OP_SIGN_SIGNED,16,OP_TYPE_INT)
        else:
            stype = (OP_SIGN_SIGNED,16,OP_TYPE_INT)
            self.modifiers.appond(".??2?")
        
        self.modifiers.append(".lo")
        
        
        rev1 = rev2 = False
        if self.fullinst:
            rev1 = self.bits(1,0x04000000)
            rev2 = self.bits(1,0x08000000)
        
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1, flip=rev2))
        self.src_operands.append(self.decode_operand(stype, 3))
        # Constant bits apply to operand 3, not 4
        self.src_operands.append(self.decode_operand(dtype, 4, reg=True, flip=rev1))

class mad24(Instruction):
    """24 bit precision multiplication+addition instruction"""
    def decode(self):
        super(mad24,self).decode()
        
        carry = self.bits(1,0x0c000000)
        if carry == 0:
            self.base = "mad24"
        elif carry == 3:
            self.base = "mad24c1" # carry from p1
        else:
            self.base = "mad24c??" # carry from p0? p2?
 
        if self.subop in [3,4,5]:
            self.modifiers.append(".lo")
        else: # 6,7
            self.modifiers.append(".hi")
        if self.subop == 5:
            self.modifiers.append(".sat")
        if self.subop in [4,5,7]:
            type = (OP_SIGN_SIGNED,32,OP_TYPE_INT)
        else: # 3,6
            type = (OP_SIGN_UNSIGNED,32,OP_TYPE_INT)
            
        rev1 = rev2 = False
        if self.fullinst:
            rev1 = self.bits(1,0x04000000)
            rev2 = self.bits(1,0x08000000)
            
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1, flip=rev2))
        self.src_operands.append(self.decode_operand(type, 3))
        # Constant bits apply to operand 3, not 4
        self.src_operands.append(self.decode_operand(type, 4, reg=True, flip=rev1))

class flop(Instruction):
    """Handler for rcp,rsqrt,lg2,sin,cos,ex2"""
    def decode(self):
        super(flop,self).decode()
        names = {0:"rcp",2:"rsqrt",3:"lg2",4:"sin",5:"cos",6:"ex2"}
        self.base = names[self.subop]
        
        type = self.default_oper_type(OPER_FLOAT)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))

class cvt0(Instruction):
    """Handler for integer to integer conversion instruction"""
    def decode(self):
        super(cvt0,self).decode()
        self.base = "cvt"
        if self.subop == 1:
            self.modifiers.append(".neg")
        try:
            stype = cvti_types[self.subsubop&7]
        except LookupError:
            stype = cvti_types[0]
            self.unknown_subsubop()

        rounding = (self.subsubop & 0x18) >> 3
        self.modifiers.append(cvt_rops[rounding])
       
        abssat = (self.subsubop & 0x60) >> 5
        self.modifiers.append(abssat_ops[abssat])
        
        dtype = self.default_oper_type(OPER_PSIZE)        
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))

class cvt2(Instruction):
    """Handler for integer to float conversion instruction"""
    def decode(self):
        super(cvt2,self).decode()
        self.base = "cvt"
        try:
            stype = cvti_types[self.subsubop&7]
        except LookupError:
            stype = cvti_types[0]
            self.unknown_subsubop()
        # Rounding mode
        self.modifiers.append(cvt_rops[(self.subsubop&0x18)>>3])
        if (self.subsubop&~0x1F): # paranoia
            self.unknown_subsubop()
            
        #if not self.bits(1,0x04000000):
        #    self.warnings.append("unknown bit not set")
        
        dtype = self.default_oper_type(OPER_VARFLOAT)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))

class satf(Instruction):
    """Handler for conversion instruction. This instruction 
    converts integer to float, but in a specific way:
    
    0 is converted to 0.0, 0xFFFFFFFF is converted to 1.0
    """
    def decode(self):
        super(satf,self).decode()
        self.base = "cvt"
        self.modifiers.append(".scale")
        # no subsubop
        
        stype = (OP_SIGN_UNSIGNED,32,OP_TYPE_INT) 
        dtype = (OP_SIGN_NONE,32,OP_TYPE_FLOAT)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))

class cvt4(Instruction):
    """Handler for float to integer conversion instruction"""
    def decode(self):
        super(cvt4,self).decode()
        self.base = "cvt"

        if (self.subsubop&7)==1:
            size = 32
        elif (self.subsubop&7)==0:
            size = 16
        else:
            size = 16
            self.unknown_subsubop() # Unknown conversion, fill this in
        # Rounding mode
        self.modifiers.append(cvt_rops[(self.subsubop&0x18)>>3]+"i")
        if (self.subsubop&~0x1F): # paranoia
            self.unknown_subsubop()
        
        stype = (OP_SIGN_NONE, size, OP_TYPE_FLOAT)
        dtype = self.default_oper_type(OPER_PSIZE)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))

class cvt6(Instruction):
    """Handler for float to float conversion instruction"""
    def decode(self):
        super(cvt6,self).decode()
        self.base = "cvt"
        
        if self.bits(1, 0x00400000):
            stype = self.default_oper_type(OPER_FLOAT64)
        elif (self.subsubop&7)==0:
            stype = self.default_oper_type(OPER_FLOAT16)
        elif (self.subsubop&7)==1:
            stype = self.default_oper_type(OPER_FLOAT32)
        else:
            self.unknown_subsubop() # Unknown conversion, fill this in

        if self.bits(1,0x08000000):
            # Integer rounding mode
            self.modifiers.append(cvt_rops[(self.subsubop>>3)&3]+"i")
        else:
            # Float rounding mode
            self.modifiers.append(cvt_rops[(self.subsubop>>3)&3])
            
        if (self.subsubop&32):
            self.modifiers.append(".sat")
        if (self.subsubop&64):
            self.modifiers.append(".abs")
            
        # all bits of subsubop are accounted for
        dtype = self.default_oper_type(OPER_VARFLOAT)
        
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))

class deltaf(Instruction):
    """Add a very small amount to a float"""
    def decode(self):
        super(deltaf,self).decode()
        self.base = "delta"
        
        if self.subsubop != 1:
            self.unknown_subsubop()
            
        type = self.default_oper_type(OPER_FLOAT)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))

class addf(Instruction):
    """Add two floating point operands"""
    def decode(self):
        super(addf,self).decode()
            
        # This determines the size of the operand (16 or 32 bit) for
        # integer instructions. For floating point instructions it 
        # has a different meaning.
        if self.fullinst:
            rev1 = self.bits(1,0x04000000)
            rev2 = self.bits(1,0x08000000)
        else:
            #rev1 = self.bits(0,0x00200000)
            rev1 = self.bits(0,0x00008000)
            rev2 = self.bits(0,0x00400000)
        
        #if self.bits(0,0x00400000):
        #    self.base = "sub" # d,a,b -> d = a-b
        #else:
        self.base = "add"
            
        # Rounding mode
        # lower bits of operand 3 is rounding op
        if self.fullinst:
            rop = self.bits(0,0x00030000)
        else:
            rop = 0
        self.modifiers.append(cvt_rops[rop])

        if self.subop == 0x3:
            type = self.default_oper_type(OPER_FLOAT64)
        else:
            type = self.default_oper_type(OPER_FLOAT32)

        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1, flip=rev1))
        if self.fullinst:
            # oper 2,1,4
            self.src_operands.append(self.decode_operand(type, 4, flip=rev2))
        else:
            # oper 2,1,3
            self.src_operands.append(self.decode_operand(type, 3, flip=rev2))

class setf(Instruction):
    """Handler for set (float)"""
    def decode(self):
        super(setf,self).decode()
        
        self.base = "set"
        self.modifiers.append(lookup(logic_ops, self.subsubop))

        dtype = self.default_oper_type(OPER_PSIZE)

        if self.subop == 0x7:
            stype = self.default_oper_type(OPER_FLOAT64)
        else:
            stype = self.default_oper_type(OPER_FLOAT32)

        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(stype, 1))
        self.src_operands.append(self.decode_operand(stype, 3))

class fmaxmin(Instruction):
    """Handler for max,min (float)"""
    def decode(self):
        super(fmaxmin,self).decode()
        
        names = {4:"max",5:"min"}
        self.base = names[self.subop]

        type = self.default_oper_type(OPER_FLOAT)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        self.src_operands.append(self.decode_operand(type, 3))


class flop6(Instruction):
    """Handler for ??. Probably a floating point normalization instruction."""
    def decode(self):
        super(flop6,self).decode()
        # I don't know what this operation does, but is executed before sin, cos, and lg2
        #names = {6:"fpre"}
        #self.base = names[self.subop]
        #self.base = "fpre"
        self.base = "pre"
        subnames = {
            0:".sin", # reduction to -PI, PI
            1:".ex2"  # reduction for exp2
        }
        
        try:
            self.modifiers.append(subnames[self.subsubop])
        except KeyError:
            self.unknown_subsubop()
        
        type = self.default_oper_type(OPER_FLOAT)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))

class mulf(Instruction):
    """Multiply two floating point operands"""
    def decode(self):
        super(mulf,self).decode()
            
        self.base = "mul"
        subsubop = self.subsubop
        if subsubop == None:
            subsubop = 0 # Set default 
        self.modifiers.append(cvt_rops[subsubop&3])
        if (subsubop&~3):
            self.unknown_subsubop()

        if self.fullinst:
            rev1 = self.bits(1,0x04000000)
            rev3 = self.bits(1,0x08000000)
        else:
            # 0,0x00200000?
            rev3 = self.bits(0,0x00400000)
            rev1 = self.bits(0,0x00008000)
        if self.subop == 0x4:
            type = self.default_oper_type(OPER_FLOAT64)
        else:
            type = self.default_oper_type(OPER_FLOAT32)
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1, flip=rev1))
        self.src_operands.append(self.decode_operand(type, 3, flip=rev3))

class slct(Instruction):
    """Select one of two operands based on the third"""
    def decode(self):
        super(slct,self).decode()
            
        self.base = "slct"

        type = self.default_oper_type(OPER_PSIZE) # what type goes here?
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1))
        self.src_operands.append(self.decode_operand(type, 3))
        self.src_operands.append(self.decode_operand(type, 4))

class logic(Instruction):
    """Various bitwise logical operations"""
    def decode(self):
        super(logic,self).decode()
        
        try:
            self.base = d0_ops[self.subsubop]
        except LookupError:
            self.unknown_subsubop()

        type = self.default_oper_type(OPER_PSIZE3) 
        self.dst_operands.append(self.decode_operand(type, 2))
        if self.base != "not":
            # Not has only second source operand
            self.src_operands.append(self.decode_operand(type, 1))
        
        
        self.src_operands.append(self.decode_operand(type, 3))

class ldofs0(Instruction):
    """Set offset pointer0 to an immediate value"""
    def decode(self):
        super(ldofs0, self).decode()
        src = self.offset_reg()
        if src:
            # We can add something to an offset register by
            # providing an offset register as source for this instruction
            self.base = "add"
        else:
            self.base = "mov"
        
        type = (OP_SIGN_NONE,32,OP_TYPE_INT)
        self.dst_operands.append(self.decode_operand(type, 2, doffset=True))
        if src:
            self.src_operands.append(Operand(OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE, src))
        self.src_operands.append(self.decode_operand(type, 5))

class ld_op(Instruction):
    """Memory load operation"""
    def decode(self):
        super(ld_op, self).decode()
        
        if self.subsubop != 0x0:
            self.unknown_subsubop()
        xx = self.bits(0,0x003F0000)
        
        self.base = "mov"
        if self.subop == 0x2:
            #self.modifiers.append(".local")
            if xx != 0:  # memory segment?
                self.modifiers.append("seg?%i" % xx)
            indirection = OP_INDIRECTION_LOCAL
        else: # 0x4
            
            #self.modifiers.append(".global")
            if xx != 14: # memory segment
                self.modifiers.append("seg?%i" % xx)
            indirection = OP_INDIRECTION_GLOBAL
        
        type = self.default_oper_type(OPER_MSIZE)
        # XXX in some cases, it seems this is the other way around
        self.dst_operands.append(self.decode_operand(type, 2, regscale=False))
        self.src_operands.append(self.decode_operand(type, 1, indir=indirection, msize_bits=False))

class st_op(Instruction):
    """Memory store operation"""
    def decode(self):
        super(st_op, self).decode()
        
        if self.subsubop != 0x0:
            # Memory segment?
            self.unknown_subsubop()
        xx = self.bits(0,0x003F0000)
        
        self.base = "mov"
        if self.subop == 0x3:
            #self.modifiers.append(".local")
            if xx != 0:  # memory segment?
                self.modifiers.append("seg?%i" % xx)
            indirection = OP_INDIRECTION_LOCAL
        else: # 0x5
            #self.modifiers.append(".global")
            if xx != 14: # memory segment
                self.modifiers.append("seg?%i" % xx)
            indirection = OP_INDIRECTION_GLOBAL
        
        type = self.default_oper_type(OPER_MSIZE)
        self.dst_operands.append(self.decode_operand(type, 1, indir=indirection, msize_bits=False))
        self.src_operands.append(self.decode_operand(type, 2, regscale=False))
        
class atomic(Instruction):
    """Memory atomic operation"""
    def decode(self):
        super(atomic, self).decode()
        
        self.base = "atom"
        if self.subop == 0x7:
            self.modifiers.append(".out")

        self.modifiers.append(lookup(atomic_ops,self.bits(1,0x0000003C)))
            
        indirection = OP_INDIRECTION_GLOBAL
        type = self.default_oper_type(OPER_MSIZE)
        self.dst_operands.append(self.decode_operand(type, 1, indir=indirection))
        self.src_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 3))

class fmad(Instruction):
    """Handler for multiply add (float)"""
    def decode(self):
        super(fmad,self).decode()

        #if not self.bits(0,0x01000000):
        if not self.fullinst:
            self.base = "mac"  # no operand 4 when operand 3 is an immediate
        else:
            self.base = "mad"
        
        # this is completely mad, as these bits overlap
        # with the 'constant segment 1' bit
        # -> when any operand is from constant mem, ignore rounding mode
        if self.fullinst and self.bits(0,0x01800000) == 0:
            rop = self.bits(1,0x00c00000)
        else:
            rop = 0
        self.modifiers.append(cvt_rops[rop])

        if self.fullinst:
            rev1 = self.bits(1,0x04000000)
            rev2 = self.bits(1,0x08000000)
        else:
            # this is wrong
            # self.bits(0,0x00400000)
            #  this is wrong too:
            # rev1 = self.bits(0,0x00200000)
            rev1 = 0
            rev2 = self.bits(0,0x00008000)
		
        if self.subop == 0x2:
            type = self.default_oper_type(OPER_FLOAT64)
        else:
            type = self.default_oper_type(OPER_FLOAT32)
        
        # XXX can't have an indirection (ofs reg) in both op1 and op3
        self.dst_operands.append(self.decode_operand(type, 2))
        self.src_operands.append(self.decode_operand(type, 1, flip=rev1))
        self.src_operands.append(self.decode_operand(type, 3))
        if self.fullinst:
            self.src_operands.append(self.decode_operand(type, 4, flip=rev2)) #reg=True, 

class tex(Instruction):
    """Handler for texture sampling instruction"""
    def decode(self):
        super(tex,self).decode()
        
        if self.subop == 0x0:
            self.base = "tex"
        elif self.subop == 0x3:
            self.base = "txq"
        op =  self.bits(0,0x01C00000)
        # Dimensionality
        args = 4
        if op in [0,4]:
            self.modifiers.append(".1d")
            args = 1
        if op in [1,6]:
            self.modifiers.append(".2d")
            args = 2
        if op in [2,7]:
            if self.bits(0,0x08000000):
                self.modifiers.append(".cube")
            else:
                self.modifiers.append(".3d")
            args = 3
        
        (ssign,stype) = (OP_SIGN_NONE,OP_TYPE_INT)
        if op in [4,6,7]:
            (ssign,stype) = (OP_SIGN_SIGNED, OP_TYPE_INT)
        elif op in [0,1,2]:
            (ssign,stype) = (OP_SIGN_NONE, OP_TYPE_FLOAT)

        if op in [3,5]: # This one is unknown
            self.modifiers.append(".%i" % op)

        if self.fullinst and not self.bits(1,0x00000004):
            self.warnings.append("unknown bit not set")
            
        # Calculate destination bitfield and source/destination registers
        value2 = self.bits(0,0x000001FC) # reg id
        if self.fullinst:
            bitfield = self.bits(0,0x06000000)|self.bits(1,0x0000c000)
        else:
            if not (value2 & 0x40):
                # XXX this changed bit field
                self.warnings.append("not (value2 & 0x40)")
            bitfield = 0xF
            value2 &= 0x3F
            
        src_set = [value2+x for x in xrange(0,args)]
        count = 0
        dst_set = []
        for x in xrange(0,4):
            if bitfield&(1<<x):
                dst_set.append(value2+count)
                count += 1
            else:
                dst_set.append(-1)

        self.dst_operands.append(Operand(value=dst_set, source=OP_SOURCE_REGSET, type=OP_TYPE_INT, sign=OP_SIGN_NONE))
        # These two are the texture
        value1 = self.bits(0,0x00007E00)
        value3 = self.bits(0,0x003E0000)
        self.src_operands.append(Operand(value=value1, source=OP_SOURCE_TEXTURE, type=OP_TYPE_TEXTURE))
        if value1 != value3:
            self.warnings.append("value3 is %i" % value3)
        self.src_operands.append(Operand(value=src_set, source=OP_SOURCE_REGSET, type=stype, sign=ssign))


        
class nop(Instruction):
    """The most boring instruction"""
    predicated = False
    def decode(self):
        super(nop, self).decode()
        
        self.base = "nop"

class unk80(Instruction):
    """Load fragment shader interpolant
    oper1 = mode? 3,4 or 0
    oper1 = normalization factor in register (?)
    oper3 = source (output of vertex shader)
    oper4 = always 8
    """
    predicated = False
    def decode(self):
        super(unk80, self).decode()
        
        self.base = "interp"
        dtype = (OP_SIGN_NONE,32,OP_TYPE_FLOAT)
        self.dst_operands.append(self.decode_operand(dtype, 2))
        self.src_operands.append(self.decode_operand(dtype, 1))
        self.src_operands.append(self.decode_operand(dtype, 3, imm=True))
        if self.fullinst:
            self.src_operands.append(self.decode_operand(dtype, 4, imm=True))

