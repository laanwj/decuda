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
# Bits known 
# inst[0]:  
#   0xF0000000 op
#   0x0C000000 offset register bit 0 and 1
#   0x02000000 offset to immediate (0), offset to register (1)
#   0x01000000 oper3 from constant
#   0x00800000 oper2 from parameter
#   0x00400000 sign of oper3 (short instruction)
#   0x003F0000 oper3 (full instruction)
#   0x001F0000 oper3 (short instruction or immediate)
#   0x00008000 sign of oper1 (short instruction)
#   0x0000FE00 oper1
#   0x000001FC oper2
#   0x00000002 0=normal, 1=system (flow control)
#   0x00000001 0=32bit, 1=64 bit
# inst[1]:  
#   0xE0000000 subop
#   0x08000000 signed/unsigned
#   0x04000000 32/16
#   0x01000000 oper3 is immediate
#   0x03C00000 segment, on constant loading
#   0x00E00000 type, on load instructions
#   0x00800000 const segment bit 1
#   0x00400000 oper3/4 from constant in segment 1
#   0x00200000 oper2 from parameter
#   0x001FC000 oper4 or sub-sub op
#   0x00003000 predicate to act on
#   0x00000800 ??
#   0x00000400 ?? (usually set, unless predicated)
#   0x00000200 execute on pred
#   0x00000100 execute on !pred
#   0x00000080 ?? (usually set, unless predicated)
#   0x00000040 set predicate
#   0x00000030 predicate to set
#   0x0000003C atomic op (cannot set predicate)
#   0x00000008 also related tot set predicate
#   0x00000004 offset register bit 2
#   0x00000003 marker (0=normal,1=end,2???,3=immediate)

# Flow control ops (defined in Flowcontrol.py)
sys_ops = [
# 0,  1,    2,    3
Exit, Bra,  Bra,  Exit, 
# 4,  5,    6,    7
Bra,  Exit, None, None, 
# 8,  9,    A,    B
Sync, Exit, Bra,  None, 
# C,  D,    E,    F
None, None, None, None, 
]

# Subop matrix (defined in Opcodes.py)
ops = [
# 00      01      02      03      04      05      06      07
[None,   t01,    t02,   ldgpu,  None,    t05,    ldofs1, stsha],
# 10      11      12      13      14      15      16      17
[ld,     ldconst,ldshar,None,   None,   None,   None,   None],
# 20      21      22      23      24      25      26      27
[add,    None,   None,   None,   None,   None,   None,   None],
# 30      31      32      33      34      35      36      37
[neg,    None,   None,   m3x,    m3x,    m3x,    m3x,    m3x],
# 40      41      42      43      44      45      46      47
[mul24,  None,   None,   None,   None,   None,   None,   None],
# 50      51      52      53      54      55      56      57
[sad,    None,   None,   None,   None,   None,   None,   None],
# 60      61      62      63      64      65      66      67
[mad16,  mad16,  None,   mad24,  mad24,  mad24,  mad24,  mad24],
# 70      71      72      73      74      75      76      77
[None,   None,   None,   None,   None,   None,   None,   None],
# 80      81      82      83      84      85      86      87
[unk80,  None,   None,   None,   None,   None,   None,   None],
# 90      91      92      93      94      95      96      97
[flop,   None,   flop,   flop,   flop,   flop,   flop,   None],
# A0      A1      A2      A3      A4      A5      A6      A7
[cvt0,   cvt0,   cvt2,   satf,   cvt4,   None,   cvt6,   deltaf],
# B0      B1      B2      B3      B4      B5      B6      B7
[addf,   None,   None,   setf,   fmaxmin,fmaxmin,flop6,  None],
# C0      C1      C2      C3      C4      C5      C6      C7
[mulf,   None,   slct,   None,   None,   None,   None,   None],
# D0      D1      D2      D3      D4      D5      D6      D7
[logic,  ldofs0, ld_op,  st_op,  ld_op,  st_op,  atomic, atomic],
# E0      E1      E2      E3      E4      E5      E6      E7
[fmad,   None,   fmad,   addf,   mulf,   None,   None,   setf],	# ???
# F0      F1      F2      F3      F4      F5      F6      F7
[tex,    None,   None,   tex,    None,   None,   None,   nop],
]


class Disassembler(object):
    def decode(self, address, inst):
        """
        Find the type of an instruction, and call for it to be decoded.
        """
        i = Instruction()
        i.address = address
        i.inst = inst
        i.decode()
        
        # Pass on to specific op
        if i.system:
            decoder = sys_ops[i.op]
        else:
            decoder = ops[i.op][i.subop]
        if decoder == None:
            # unknown instruction
            # system, op, subop
            return i
        else:
            # Use specific decoder
            i = decoder(i)
            try:
                i.decode()
            except DecodeError, e: # Error decoding instruction
                i.warnings.append(e.message)
            return i

