#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan, 2007
from cStringIO import StringIO
from Operand import *
from Instruction import *

class Bra(Instruction):    
    def decode(self):
        super(Bra,self).decode()
        
        if self.op == 0x1:
            self.base = "bra"
        elif self.op == 0x2:
            self.base = "call"
        elif self.op == 0x4:
            self.base = "breakaddr"
            self.predicated = False
        else: # 0xA
            self.base = "join"
            self.predicated = False

        # Branch instruction, divide address for convenience
        # I know the address is longer, as CUDA can address up to
        # 2Mb of kernel instructions, but I have never been 
        # able to generate a kernel this big without crashing the
        # ptxas. Probably, the higher part is in inst[1].
        addr = self.bits(0,0x0FFFFE00)
        # Shift right by two as instructions are in 4 byte bound
        if addr&3:
            self.warnings.append("Non-aligned target address for branch!")

        self.dst_operands.append(Operand(
            OP_TYPE_INT, OP_SIGN_UNSIGNED, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_CODE, addr))

class Exit(Instruction):    
    def decode(self):
        super(Exit,self).decode()
        
        if self.op == 0x0:
            self.base = "kil" # kill fragment
        elif self.op == 0x3:
            self.base = "return"
        elif self.op == 0x5:
            self.base = "break"
        else: #if self.op == 0x9:
            self.base = "trap"
            self.predicated = False

class Sync(Instruction):
    predicated = False
    def decode(self):
        super(Sync,self).decode()
        
        self.base = "bar.sync"
        op = self.bits(0, 0x06000000)
        bar = self.bits(0, 0x01e00000)
        arg = self.bits(0, 0x001ffe00)
        if op != 3 or arg != 0xfff:
            self.warnings.append("unknown barrier op %01x arg %03x", op, arg)

        self.dst_operands.append(Operand(
            OP_TYPE_INT, OP_SIGN_UNSIGNED, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_NONE, bar))
