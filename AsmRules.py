#!/usr/bin/python
# sm1_1 (G80) (dis)assembler 
# Wladimir J. van der Laan, 2007
#   the big monster table that aims to represent the G8x instruction set
# The aim of this assembler is to facilitate automatic code generation, so we ignore all kinds of fluff
# we can ignore some modifiers, like .global, .local, .shared, .const on ld as they
# are implied by operands. Furthermore, ld, st and mov are the same operation, just from a different
# perspective. Move is move, the user is free to choose how he wants to call it.

from Operand import *
from Instruction import *
from Util import wraplist
from AsmConstants import *
# All constant segments
ALL_CONST = range(OP_INDIRECTION_CONST0, OP_INDIRECTION_CONST15+1)
# All operand sizes
ALL_SIZES = [8,16,32,64,128]

# operand signs
class _s:
    B = OP_SIGN_NONE
    S = OP_SIGN_SIGNED
    U = OP_SIGN_UNSIGNED
    X = [OP_SIGN_NONE, OP_SIGN_SIGNED, OP_SIGN_UNSIGNED]
    US = [OP_SIGN_SIGNED, OP_SIGN_UNSIGNED]


_regw = {16:OP_SOURCE_HALF_REGISTER, 32:OP_SOURCE_REGISTER}
_regwo = {16:OP_SOURCE_HALF_OUTPUT_REGISTER, 32:OP_SOURCE_OUTPUT_REGISTER}

# integer
def i_reg(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, [_regw[x] for x in wraplist(width)], OP_INDIRECTION_NONE)
# register or output register
def i_oreg(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, [_regw[x] for x in wraplist(width)] + [_regwo[x] for x in wraplist(width)], OP_INDIRECTION_NONE)
def i_s(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_SHARED)
def i_c(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, OP_SOURCE_IMMEDIATE, ALL_CONST)
def i_sr(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, OP_SOURCE_REGISTER, OP_INDIRECTION_SHARED)
def i_cr(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, OP_SOURCE_REGISTER, ALL_CONST)
def i_imm(sign, width=[16,32]):
    return (OP_TYPE_INT, sign, width, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_NONE)

# float (width defaults to 32 bit, no signedness)
# XXX mov instructions and other 'data-neutral' instructions should not care
# whether arguments are float or int
def f_reg(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, [_regw[x] for x in wraplist(width)], OP_INDIRECTION_NONE)
def f_oreg(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, [_regw[x] for x in wraplist(width)] + [_regwo[x] for x in wraplist(width)], OP_INDIRECTION_NONE)
def f_s(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_SHARED)
def f_c(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, OP_SOURCE_IMMEDIATE, ALL_CONST)
def f_sr(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, OP_SOURCE_REGISTER, OP_INDIRECTION_SHARED)
def f_cr(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, OP_SOURCE_REGISTER, ALL_CONST)
def f_imm(width=32):
    return (OP_TYPE_FLOAT, OP_SIGN_NONE, width, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_NONE)

# XXX there is only one set of offset register bits, does that get enabled for both operands?

# bitfield list
BF_ALWAYS       = None            # bitfield that is always 0xFFFFFFFF
BF_OP           = (0, 0xf0000000)
BF_FLOWCTL_ADDR = (0, 0x0FFFFE00) # address, for flow control inst
BF_OFFSET01     = (0, 0x0C000000) # bit 0 and 1 of offset register
BF_OFFSET_TO    = (0, 0x02000000) # offset to immediate (0), offset to register (1)
BF_OPER1_SHARED_H=(0, 0x01000000) # oper1 from shared (half instruction)
BF_OPER4_CONST  = (0, 0x01000000) # oper4 from constant (full instruction)
BF_OPER3_CONST  = (0, 0x00800000) # oper3 from constant
BF_ALT          = (0, 0x00400000) # alternative instruction (add->sub etc)
BF_SYNC_OP      = (0, 0x06000000) # sync op (3)
BF_SYNC_BAR     = (0, 0x01e00000) # sync barrier 0-15
BF_SYNC_ARG     = (0, 0x001ffe00) # sync ??
BF_FLOW_ADDR    = (0, 0x0FFFFE00) # address, flow control
BF_OPER5        = (0, 0x003FFE00) # merge oper 1 and 3
BF_OPER3_F      = (0, 0x003F0000) # oper3 (full instruction or immediate)
BF_OPER3_HH     = (0, 0x00200000) # oper3 high bit (special, when half instruction)
BF_OPER3_L      = (0, 0x001F0000) # oper3 (half instruction)
BF_FLOP_ROP     = (0, 0x00030000) # rounding operation for some floating point ops
BF_OPER1        = (0, 0x0000FE00) # oper1 (usually source register, full instruction)
BF_OPER1_SH     = (0, 0x0000C000) # operand size and sign for shared memory access (full instruction)
BF_OPER1_H      = (0, 0x00008000) # oper1 high bit (32 bit flag, for half instruction)
BF_OPER1_L      = (0, 0x00007E00) # oper1 low bits (half instruction)
BF_OPER1_LSH    = (0, 0x00006000) # operand size and sign for shared memory access (half instruction)
BF_OPER1_SL     = (0, 0x00003E00) # memory location number for shared memory access (full instruction)
BF_OPER1_LSL    = (0, 0x00001E00) # memory location number for shared memory access (half instruction)
BF_OPER2        = (0, 0x000001FC) # oper2 (usually destination)
BF_FLOWCTL      = (0, 0x00000002)
BF_INSTWIDTH    = (0, 0x00000001) # 0=32bit, 1=64bit
BF_SUBOP        = (1, 0xe0000000)
BF_PSIZE_SIGN   = (1, 0x08000000) # 0=unsigned, 1=signed
BF_FLOP_FLIP2   = (1, 0x08000000)
BF_CVT_I        = (1, 0x08000000) # cvt integer rounding
BF_OPER3_H      = (1, 0x0FFFFFFC) # immediate value, high bits (extended operand 3)
BF_PSIZE_WIDTH  = (1, 0x04000000) # 0=16 bit, 1=32 bit
BF_FLOP_FLIP1   = (1, 0x04000000)
BF_MSIZE        = (1, 0x00E00000) # type, for memory load/store instructions
BF_STSHA_SZ     = (1, 0x00E00000) # store shared (0=16 bit, 1=32 bit, 2=8 bit)
BF_CONSTSEG_X   = (1, 0x03C00000) # full constant segment, only used in some instructions (or in all, unsure)
BF_CONSTSEG     = (1, 0x00C00000) # constant segment
BF_MAD_ROP      = (1, 0x00C00000) # mad rounding op
BF_OPER1_SHARED_F=(1, 0x00200000) # oper1 shared (full instruction)
BF_SUBSUBOP     = (1, 0x001fc000) # subsubop or oper4
BF_OPER4        = (1, 0x001fc000) # subsubop or oper4
BF_CVT_ABSSAT   = (1, 0x00180000) # absolute/saturation
BF_OPER3_IMM    = (1, 0x00100000) # operand 3 is immediate value, but only if no oper4 present?
BF_CVT_RM       = (1, 0x00060000) # rounding mode for cvt
BF_CVTI_TYPE    = (1, 0x0001c000) # source type for cvt
BF_SUBSUB_ROP   = (1, 0x0000c000) # rounding operation for some floating point ops
BF_LOP_INV2     = (1, 0x00020000) # logic operation (invert argument 2)
BF_LOP_INV1     = (1, 0x00010000) # logic operation (invert argument 1)
BF_LOP_OP       = (1, 0x0000c000) # logic operation in 0xd0 instruction
BF_MUL24_SRC32  = (1, 0x00010000)
BF_MUL24_SIGN   = (1, 0x00008000)
BF_MUL24_HI     = (1, 0x00004000)
BF_PRED_IN      = (1, 0x00003000) # predicate
BF_CC           = (1, 0x00000F80) # condition code
BF_SET_PRED     = (1, 0x00000040) # instruction sets predicate
BF_ATOMIC       = (1, 0x0000003C) # atomic operation
BF_PRED_OUT     = (1, 0x00000030) # predicate to set if BF_SETPRED
BF_OUTPUTREG    = (1, 0x00000008) # output to output register (if also oper2==0x7f, ignore output)
BF_OFFSET2      = (1, 0x00000004) # bit 2 of offset register #
BF_MARKER       = (1, 0x00000003) # type of instruction 0 normal 1 end 2 join 3 immediate



# Instruction types
def wide_op(x):
    """64 bit operation. Adds the modifiers .end and .join as well."""
    return [(BF_OP, IMM, x>>4), (BF_SUBOP, IMM, x&0xF)] + bit(BF_INSTWIDTH) + [(BF_MARKER, MODIFIER, ".end", 1), (BF_MARKER, MODIFIER, ".join", 2)]

def half_op(x):
    """32 bit instruction"""
    return [(BF_OP, IMM, x>>4)]

def imm_op(x):
    """64 bit immediate operation"""
    return [(BF_OP, IMM, x>>4), (BF_SUBOP, IMM, x&0xF)] + bit(BF_INSTWIDTH) + [(BF_MARKER, IMM, 3)]

def s_op(x):
    """64 bit flow control operation"""
    return [(BF_OP, IMM, x>>4), (BF_SUBOP, IMM, x&0xF)] + bit(BF_INSTWIDTH) + bit(BF_FLOWCTL) + [(BF_MARKER, MODIFIER, ".end", 1), (BF_MARKER, MODIFIER, ".join", 2)]


# Easy bit fields    
def bit(x):
    """set a single bit"""
    return [(x, IMM, 1)]

def subsubop(x):
    """sub-sub operation"""
    return [(BF_SUBSUBOP, IMM, x)]


# default destination, almost all instructions have this
dest = [(BF_OPER2, DST1, VALUE)]
dest_oreg = dest + [(BF_OUTPUTREG, DST1, IS_OUTREG)]
pred_out = [(BF_SET_PRED, PRED_OUT, PRESENT), (BF_PRED_OUT, PRED_OUT, VALUE)]
pred_in  = [(BF_PRED_IN, PRED_IN, VALUE), (BF_CC, PRED_IN, CC)]

# constant segment for src2 (full instruction)
def constseg(src):
    return [(BF_CONSTSEG, src, CONSTSEG)] 
# half instruction
def constseg_l(src):
    return [(BF_OPER3_HH, src, CONSTSEG)] 
# offset bits (wide instruction)
def offset_bits(src):
    return [(BF_OFFSET01, src, OFFSET, 3), (BF_OFFSET2, src, OFFSET, 4), (BF_OFFSET_TO, src, OFFSET_INC)]
# offset bits (half instruction)
def offset_bits_l(src):
    return [(BF_OFFSET01, src, OFFSET, 3), (BF_OFFSET_TO, src, OFFSET_INC)]
    
# source routing
#   registers
def reg_1(x):
    return [(BF_OPER1, x, VALUE)]
def reg_1l(x):
    return [(BF_OPER1_L, x, VALUE)]
def reg_3(x):
    return [(BF_OPER3_F, x, VALUE)]
def reg_3l(x):
    return [(BF_OPER3_F, x, VALUE)]
def reg_4(x):
    return [(BF_OPER4, x, VALUE)]    

#  constants
def const_3(x):
    return [(BF_OPER3_F, x, VALUE_ALIGN)] + bit(BF_OPER3_CONST) + constseg(x) + offset_bits(x)
def const_4(x):
    return [(BF_OPER4, x, VALUE_ALIGN)] + bit(BF_OPER4_CONST) + constseg(x) + offset_bits(x)
def const_3l(x):
    return [(BF_OPER3_L, x, VALUE_ALIGN)] + bit(BF_OPER3_CONST) + constseg_l(x) + offset_bits_l(x)
    
#  shared
def shared_1(x):
    return [(BF_OPER1_SH, x, SHTYPE), (BF_OPER1_SL, x, VALUE_ALIGN)] + bit(BF_OPER1_SHARED_F) + offset_bits(x)
def shared_1l(x):
    return [(BF_OPER1_LSH, x, SHTYPE), (BF_OPER1_LSL, x, VALUE_ALIGN)] + bit(BF_OPER1_SHARED_H) + offset_bits_l(x)
#  XXX is the BF_OPER1_SH/BF_OPER1_SL distinction there also if we use registers to index into the shared memory
#  pool?

#  short immediate
def imm_3(x):
    return [(BF_OPER3_F, x, VALUE)] + bit(BF_OPER3_IMM)
#  long immediate
def imm_3w(x):
    return [(BF_OPER3_F, x, VALUE, 0x3f), (BF_OPER3_H, x, VALUE, 0xffffffc0)]


# psize width
def psize_width(x):
    return [(BF_PSIZE_WIDTH, y, IS_32BIT) for y in wraplist(x)]
# psize width, short instruction
def psize_width_s(x):
    return [(BF_OPER1_H, y, IS_32BIT) for y in wraplist(x)]
# psize sign
def psize_sign(x):
    return [(BF_PSIZE_SIGN, y, IS_SIGNED) for y in wraplist(x)]

# psize width for one and two operands
psize_width1 = psize_width([DST1, SRC1])
psize_width2 = psize_width([DST1, SRC1, SRC2])

# psize width for one and two operands, short instruction
psize_width1_s = psize_width_s([DST1, SRC1])
psize_width2_s = psize_width_s([DST1, SRC1, SRC2])

# psize sign for one and two operands
psize_sign1 = psize_sign([DST1, SRC1])
psize_sign2 = psize_sign([DST1, SRC1, SRC2])

# msize (for device memory)
msize_12 = [(BF_MSIZE, DST1, GET_MSIZE), (BF_MSIZE, SRC1, GET_MSIZE)]

# standard operand description for a lot of instructions
def std_operands_1_3(sign):
    return [
    (DST1, i_oreg(sign), dest_oreg),
    (SRC1, i_reg(sign),  reg_1(SRC1)), 
    (SRC1, i_s(sign),    shared_1(SRC1)),
    (SRC2, i_reg(sign),  reg_3(SRC2)), 
    (SRC2, i_c(sign),    const_3(SRC2)),
#    (SRC2, i_imm(sign),  imm_3(SRC2))
   ]
def std_operands_f1_3():
    return [
    (DST1, f_oreg(), dest_oreg),
    (SRC1, f_reg(),  reg_1(SRC1)), 
    (SRC1, f_s(),    shared_1(SRC1)),
    (SRC2, f_reg(),  reg_3(SRC2)), 
    (SRC2, f_c(),    const_3(SRC2)),
#    (SRC2, f_imm(),  imm_3(SRC2))
   ]

# standard operand description for a lot of instructions
def std_operands_1_4(sign):
    return [
    (DST1, i_oreg(sign), dest_oreg),
    (SRC1, i_reg(sign),  reg_1(SRC1)), 
    (SRC1, i_s(sign),    shared_1(SRC1)),
    (SRC2, i_reg(sign),  reg_4(SRC2)), 
    (SRC2, i_c(sign),    const_4(SRC2)),
#    (SRC2, i_imm(sign),  src2_4_imm)
   ]
def std_operands_f1_4():
    return [
    (DST1, f_oreg(), dest_oreg),
    (SRC1, f_reg(),  reg_1(SRC1)), 
    (SRC1, f_s(),    shared_1(SRC1)),
    (SRC2, f_reg(),  reg_4(SRC2)), 
    (SRC2, f_c(),    const_4(SRC2)),
#    (SRC2, i_imm(sign),  src2_4_imm)
   ]

# standard operand description for a lot of instructions
def std_operands_1(sign):
    return [
    (DST1, i_oreg(sign), dest_oreg),
    (SRC1, i_reg(sign),  reg_1(SRC1)), 
    (SRC1, i_s(sign),    shared_1(SRC1)),
   ]

# standard operand description for a lot of instructions
def std_operands_f1():
    return [
    (DST1, f_oreg(), dest_oreg),
    (SRC1, f_reg(),  reg_1(SRC1)), 
    (SRC1, f_s(),    shared_1(SRC1)),
   ]

# standard operands (half instruction)
def std_operands_l1_3(sign):
    return [
    (DST1, i_reg(sign),  dest),
    (SRC1, i_reg(sign),  reg_1l(SRC1)), 
    (SRC1, i_s(sign),    shared_1l(SRC1)),
    (SRC2, i_reg(sign),  reg_3l(SRC2)), 
    (SRC2, i_c(sign),    const_3l(SRC2)),
    ]
def std_operands_fl1_3():
    return [
    (DST1, f_reg(),  dest),
    (SRC1, f_reg(),  reg_1l(SRC1)), 
    (SRC1, f_s(),    shared_1l(SRC1)),
    (SRC2, f_reg(),  reg_3l(SRC2)), 
    (SRC2, f_c(),    const_3l(SRC2))
   ]
def std_operands_fl1():
    return [
    (DST1, f_reg(),  dest),
    (SRC1, f_reg(),  reg_1l(SRC1)), 
    (SRC1, f_s(),    shared_1l(SRC1)),
   ]
# standard operands (imm wide instruction)
def std_operands_imm1_3(sign):
    return [
    (DST1, i_reg(sign),  dest),
    (SRC1, i_reg(sign),  reg_1l(SRC1)), 
    (SRC1, i_s(sign),    shared_1l(SRC1)),
    (SRC2, i_imm(sign),  imm_3w(SRC2)), 
    ]
def std_operands_immf1_3():
    return [
    (DST1, f_reg(),  dest),
    (SRC1, f_reg(),  reg_1l(SRC1)), 
    (SRC1, f_s(),    shared_1l(SRC1)),
    (SRC2, f_imm(),  imm_3w(SRC2)), 
   ]

# Rounding modes (float and integer)
def _rmf(x):
    return [(x, MODIFIER, ".rn", 0), (x, MODIFIER, ".rm", 1),
            (x, MODIFIER, ".rp", 2), (x, MODIFIER, ".rz", 3)]
def _rmi(x):
    return [(x, MODIFIER, ".rni", 0), (x, MODIFIER, ".rmi", 1),
            (x, MODIFIER, ".rpi", 2), (x, MODIFIER, ".rzi", 3)]

# flip operands
_flop_flip = [(BF_FLOP_FLIP1, SRC1, FLIP), (BF_FLOP_FLIP2, SRC2, FLIP)]
_set_ops = [(BF_SUBSUBOP, MODIFIER, ".fl", 0), (BF_SUBSUBOP, MODIFIER, ".lt", 1),
     (BF_SUBSUBOP, MODIFIER, ".eq", 2), (BF_SUBSUBOP, MODIFIER, ".le", 3), 
     (BF_SUBSUBOP, MODIFIER, ".gt", 4), (BF_SUBSUBOP, MODIFIER, ".ne", 5),
     (BF_SUBSUBOP, MODIFIER, ".ge", 6), (BF_SUBSUBOP, MODIFIER, ".tr", 7)]

# Mad (integer) operands
def _mad_operands(sign, width):
    return [(DST1, i_oreg(sign, 32), dest_oreg),
          (SRC1, i_reg(sign, width),  reg_1(SRC1)), 
          (SRC1, i_s(sign, width),    shared_1(SRC1)),
          (SRC2, i_reg(sign, width),  reg_3(SRC2)), 
          (SRC2, i_c(sign, width),    const_3(SRC2)),
#          (SRC2, i_imm(sign, width),  imm_3(SRC2)),   would overlap with src4
          (SRC3, i_reg(sign, 32),  reg_4(SRC3))]  
_mad_flip = [(BF_FLOP_FLIP1, SRC3, FLIP), (BF_FLOP_FLIP2, SRC1, FLIP)]

# lop (0xd0) modifiers
def _logic(x): 
    return [(BF_LOP_OP, IMM, x), (BF_LOP_INV1, SRC1, INVERT), (BF_LOP_INV2, SRC2, INVERT)]

# oper addr/label
oper_addr = [(DST1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_CODE), [(BF_FLOW_ADDR, DST1, VALUE)])]

def _always(x):
    return [(BF_ALWAYS, MODIFIER, x, 1)]

# modifiers
m_sat =  [(BF_PSIZE_SIGN, MODIFIER, ".sat", 1)]

# force short instruction, for testing
halfflag = _always(".half")

# === define a list of rules == machine description == 
# [op, length, base rule, [operand rule...]]
rules = [
# ----------------------------------------------------------------------------
# 64 bit full instructions
# saturating add
#("add", 2, wide_op(0x20) + pred_out + pred_in + m_flip + psize_width2 + bit(BF_PSIZE_SIGN) +
#   [(BF_ALWAYS, MODIFIER, ".sat", 1)], # require sat
#   std_operands_1_4(_s.S)),
# normal add .. rename these instructions?
# add          + a + b
# sub          + a - b
# subr         - a + b
# addc         + a + b + $p0.cf
#    .sat     saturate between -0x80000000 .. + 0x7FFFFFFF (signed only)
("add", 2, wide_op(0x20) + pred_out + pred_in + psize_width2 + m_sat, std_operands_1_4(_s.X)),
("sub", 2, wide_op(0x20) + pred_out + pred_in + bit(BF_ALT) + psize_width2 + m_sat, std_operands_1_4(_s.X)),
# subtract with reversed operands
# can these be .sat too by setting sign bit?
("subr", 2, wide_op(0x30) + pred_out + pred_in + psize_width2 + m_sat, std_operands_1_4(_s.X)),
# add with carry (pred_in is input predicate register from which the carry bit is taken, so 
# this instruction cannot be predicated itself)
("addc", 2, wide_op(0x30) + bit(BF_ALT) + pred_out + pred_in + psize_width2 + m_sat, std_operands_1_4(_s.X)),

# integer multiplication (dest always 32 bit)
# bit 0 "BF_MUL24_HI"
# bit 1 "BF_MUL24_SIGN"
# bit 2 "BF_MUL24_SRC32"
("mul24", 2, wide_op(0x40) + pred_out + pred_in + [
    (BF_MUL24_SRC32, SRC1, IS_32BIT), (BF_MUL24_SRC32, SRC2, IS_32BIT),
    (BF_MUL24_SIGN, DST1, IS_SIGNED), (BF_MUL24_SIGN, SRC1, IS_SIGNED), (BF_MUL24_SIGN, SRC2, IS_SIGNED), 
    (BF_MUL24_HI, MODIFIER, ".lo", 0), (BF_MUL24_HI, MODIFIER, ".hi", 1)
  ], [
    (DST1, i_oreg(_s.US, 32), dest_oreg),
    (SRC1, i_reg(_s.US, [16,32]),  reg_1(SRC1)), 
    (SRC1, i_s(_s.US, [16,32]),    shared_1(SRC1)),
    (SRC2, i_reg(_s.US, [16,32]),  reg_3(SRC2)), 
    (SRC2, i_c(_s.US, [16,32]),    const_3(SRC2))
]),

# integer mad
# 0 mad24  lo  16  unsigned
# 1 None   ?
# 2 None   ?
# 3 mad24  lo  32  unsigned
# 4 mad24  lo  32  signed
# 5 mad24  lo  32  signed sat
# 6 mad24  hi  32  unsigned
# 7 mad24  hi  32  signed
("mad24", 2, wide_op(0x60) + pred_out + pred_in + _always(".lo") + _mad_flip, _mad_operands(_s.U, 16)),
("mad24", 2, wide_op(0x63) + pred_out + pred_in + _always(".lo") + _mad_flip, _mad_operands(_s.U, 32)),
("mad24", 2, wide_op(0x64) + pred_out + pred_in + _always(".lo") + _mad_flip, _mad_operands(_s.S, 32)),
("mad24", 2, wide_op(0x65) + pred_out + pred_in + _always(".lo") + _mad_flip + _always("sat"), _mad_operands(_s.S, 32)),
("mad24", 2, wide_op(0x66) + pred_out + pred_in + _always(".hi") + _mad_flip, _mad_operands(_s.U, 32)),
("mad24", 2, wide_op(0x67) + pred_out + pred_in + _always(".hi") + _mad_flip, _mad_operands(_s.S, 32)),

# conditional set (int)
("set", 2, wide_op(0x33) + pred_out + pred_in + psize_width2 + psize_sign2 + _set_ops, std_operands_1_3(_s.US)),
# conditional set (float)
# XXX what does the sign bit do here?
("set", 2, wide_op(0xb3) + pred_out + pred_in + psize_width(DST1) + bit(BF_PSIZE_SIGN) +_set_ops, [
    (DST1, i_oreg(_s.X), dest_oreg),
    (SRC1, f_reg(),  reg_1(SRC1)), 
    (SRC1, f_s(),    shared_1(SRC1)),
    (SRC2, f_reg(),  reg_3(SRC2)), 
    (SRC2, f_c(),    const_3(SRC2)),
    (SRC2, f_imm(),  imm_3(SRC2))
   ]
),

# integer operators
("max", 2, wide_op(0x34) + pred_out + pred_in + psize_width2 + psize_sign2,  std_operands_1_3(_s.US) + [(SRC2, i_imm(_s.US),  imm_3(SRC2))]),
("min", 2, wide_op(0x35) + pred_out + pred_in + psize_width2 + psize_sign2,  std_operands_1_3(_s.US) + [(SRC2, i_imm(_s.US),  imm_3(SRC2))]),
("shl", 2, wide_op(0x36) + pred_out + pred_in + psize_width2 + psize_sign2,  std_operands_1_3(_s.US) + [(SRC2, i_imm(_s.US),  imm_3(SRC2))]),
("shr", 2, wide_op(0x37) + pred_out + pred_in + psize_width2 + psize_sign2,  std_operands_1_3(_s.US) + [(SRC2, i_imm(_s.US),  imm_3(SRC2))]),
# no operation
("nop", 2, wide_op(0xF7) + pred_in, []),
# load instructions
# from shared or register to a register
("mov", 2, wide_op(0x10) + subsubop(0xF) + pred_out + pred_in + psize_width1, [
    (DST1, i_oreg(_s.B), dest_oreg),
    (SRC1, i_reg(_s.B),  reg_1(SRC1)), 
    (SRC1, i_s(_s.B),    shared_1(SRC1)),
]),
# from constant to a register
("mov", 2, wide_op(0x11) + [(BF_SUBSUBOP, SRC1, SHTYPE)] + pred_out + pred_in + psize_width(DST1), [
    (DST1, i_oreg(_s.X), dest_oreg),
    (SRC1, i_c(_s.X),   [(BF_OPER5, SRC1, VALUE_ALIGN), (BF_CONSTSEG_X, SRC1, CONSTSEG)] + offset_bits(SRC1)), 
]),
# from shared to a register
("mov", 2, wide_op(0x12) + [(BF_SUBSUBOP, SRC1, SHTYPE)] + pred_out + pred_in + psize_width(DST1), [
    (DST1, i_oreg(_s.X), dest_oreg),
    (SRC1, i_s(_s.X),   [(BF_OPER5, SRC1, VALUE_ALIGN)] + offset_bits(SRC1)), 
]),
# from register to shared
# psize is used for source operand only
("mov", 2, wide_op(0x07) + pred_out + pred_in + psize_width(SRC1), [
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 16, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_SHARED), [(BF_STSHA_SZ, IMM, 0), (BF_OPER5, DST1, VALUE_ALIGN)]),
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_SHARED), [(BF_STSHA_SZ, IMM, 1), (BF_OPER5, DST1, VALUE_ALIGN)]),
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 8,  OP_SOURCE_IMMEDIATE, OP_INDIRECTION_SHARED), [(BF_STSHA_SZ, IMM, 2), (BF_OPER5, DST1, VALUE_ALIGN)]),
    (SRC1, i_reg(OP_SIGN_NONE), [(BF_OPER4, SRC1, VALUE)])
]),
# from predicate to register
# psize is used for desination operand
("mov", 2, wide_op(0x01) + pred_out + pred_in + psize_width(DST1), [
    (DST1, i_oreg(_s.B), dest_oreg),
    (SRC1, (OP_TYPE_PRED, OP_SIGN_NONE, 1, OP_SOURCE_PRED_REGISTER, OP_INDIRECTION_NONE), [(BF_OPER1, SRC1, VALUE)])
]),
# from offset to register (always 32 bit)
("mov", 2, wide_op(0x02) + pred_out + pred_in, [
    (DST1, i_oreg(_s.B,32), dest_oreg),
    (SRC1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE), [(BF_OFFSET01, SRC1, VALUE, 3), (BF_OFFSET2, SRC1, VALUE, 4)])
]),
# from register to predicate
# psize is used for source operand
("mov", 2, wide_op(0x05) + pred_in + pred_out + psize_width(SRC1), [
#    (DST1, (OP_TYPE_PRED, OP_SIGN_NONE, 1, OP_SOURCE_PRED_REGISTER, OP_INDIRECTION_NONE), [(BF_PRED_OUT, DST1, VALUE), (BF_SET_PRED, IMM, 1)]),
    (SRC1, i_reg(OP_SIGN_NONE), [(BF_OPER1, SRC1, VALUE)])
]),
# from internal to register (always 32 bit)
# i[0]..i[7]
("mov", 2, wide_op(0x03) + pred_out + pred_in, [
    (DST1, i_oreg(_s.B,32), dest_oreg),
    (SRC1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_INTERNAL), [(BF_OPER4, SRC1, VALUE)])
]),
# load imm to offset register
("mov", 2, wide_op(0xd1) + pred_out + pred_in, [
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE), [(BF_OPER2, DST1, VALUE)]),
    (SRC1, i_imm(_s.B,32), [(BF_OPER5, SRC1, VALUE)]),   
]),
# add imm to offset register
("add", 2, wide_op(0xd1) + pred_out + pred_in, [
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE), [(BF_OPER2, DST1, VALUE)]),
    (SRC1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE), [(BF_OFFSET01, SRC1, VALUE, 3), (BF_OFFSET2, SRC1, VALUE, 4)]),
    (SRC2, i_imm(_s.B,32), [(BF_OPER5, SRC2, VALUE)]),   
]),
# move and shift register to offset register
("movsh", 2, wide_op(0x06) + pred_out + pred_in, [
    (DST1, (OP_TYPE_INT, OP_SIGN_NONE, 32, OP_SOURCE_OFFSET_REGISTER, OP_INDIRECTION_NONE), [(BF_OPER2, DST1, VALUE)]),
    (SRC1, i_reg(_s.B,32), [(BF_OPER1, SRC1, VALUE)]),   
    (SRC2, i_imm(_s.B,32), [(BF_OPER3_F, SRC2, VALUE)]),   
]),
# global, local loads
# afaik, there is only one local segment
("mov", 2, wide_op(0xd2) + pred_out + pred_in + msize_12 + [(BF_OPER3_F, IMM, 0)], [
    (DST1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),   [(BF_OPER2, DST1, VALUE)]),
    (SRC1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_LOCAL),  [(BF_OPER1, SRC1, VALUE)]),
]),
("mov", 2, wide_op(0xd3) + pred_out + pred_in + msize_12 + [(BF_OPER3_F, IMM, 0)], [
    (DST1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_LOCAL),  [(BF_OPER1, DST1, VALUE)]),
    (SRC1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),   [(BF_OPER2, SRC1, VALUE)]),   
]),
# default global to segment 14 for now
("mov", 2, wide_op(0xd4) + pred_out + pred_in + msize_12 + [(BF_OPER3_F, IMM, 14)], [
    (DST1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),   [(BF_OPER2, DST1, VALUE)]),
    (SRC1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_GLOBAL), [(BF_OPER1, SRC1, VALUE)]),
]),
("mov", 2, wide_op(0xd5) + pred_out + pred_in + msize_12 + [(BF_OPER3_F, IMM, 14)], [
    (DST1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_GLOBAL), [(BF_OPER1, DST1, VALUE)]),
    (SRC1, (OP_TYPE_INT, _s.X, ALL_SIZES, OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),   [(BF_OPER2, SRC1, VALUE)]),   
]),
# cvt (conversion).. these are nasty
# integer to integer
# src 8 bit -> half-register
# src 16 bit -> half-register
# src 32 bit -> full-register
("cvt", 2, wide_op(0xa0) + pred_out + pred_in + psize_width(DST1) + psize_sign(DST1) + [(BF_CVTI_TYPE, SRC1, CVTI_TYPE)], [
    (DST1, i_oreg(_s.US),     dest_oreg),
    (SRC1, (OP_TYPE_INT, _s.US, [8,16], OP_SOURCE_HALF_REGISTER, OP_INDIRECTION_NONE), reg_1(SRC1)), 
    (SRC1, (OP_TYPE_INT, _s.US, [32],   OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),      reg_1(SRC1)), 
    (SRC1, i_s(_s.US, [8,16,32]),    shared_1(SRC1)),
]),
# like cvt, but negates result
# only works for signed types, it appears, for u32 and u16 destination types, it sets the output to 0
("cvt", 2, wide_op(0xa1) + pred_out + pred_in + psize_width(DST1) + psize_sign(DST1) + _always(".neg") + [(BF_CVTI_TYPE, SRC1, CVTI_TYPE)], [
    (DST1, i_oreg(_s.US),     dest_oreg),
    (SRC1, (OP_TYPE_INT, _s.US, [8,16], OP_SOURCE_HALF_REGISTER, OP_INDIRECTION_NONE), reg_1(SRC1)), 
    (SRC1, (OP_TYPE_INT, _s.US, [32],   OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),      reg_1(SRC1)), 
    (SRC1, i_s(_s.US, [8,16,32]),    shared_1(SRC1)),
]),

# integer to float
("cvt", 2, wide_op(0xa2) + pred_out + pred_in + psize_width(DST1) + [(BF_CVTI_TYPE, SRC1, CVTI_TYPE)] + _rmf(BF_CVT_RM), [
    (DST1, f_oreg([16,32]),     dest_oreg),
    (SRC1, (OP_TYPE_INT, _s.US, [8,16], OP_SOURCE_HALF_REGISTER, OP_INDIRECTION_NONE), reg_1(SRC1)), 
    (SRC1, (OP_TYPE_INT, _s.US, [32],   OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),      reg_1(SRC1)), 
    (SRC1, i_s(_s.US, [8,16,32]),    shared_1(SRC1)),
]),
# float to integer
("cvt", 2, wide_op(0xa4) + pred_out + pred_in + psize_width(DST1) + psize_sign(DST1) + [(BF_CVTI_TYPE, SRC1, IS_32BIT)] + _rmi(BF_CVT_RM), [
    (DST1, i_oreg(_s.US),     dest_oreg),
    (SRC1, f_reg([16,32]),    reg_1(SRC1)), 
    (SRC1, f_s([16,32]),      shared_1(SRC1)),
]),
# 0xa5 ??
# float to float (supports both integer and float rounding modes)
("cvt", 2, wide_op(0xa6) + pred_out + pred_in + psize_width(DST1) + [(BF_CVTI_TYPE, SRC1, IS_32BIT)] + _rmi(BF_CVT_RM) + _rmf(BF_CVT_RM) + [
        (BF_CVT_I, MODIFIER, ".rn", 0), (BF_CVT_I, MODIFIER, ".rm", 0),
        (BF_CVT_I, MODIFIER, ".rp", 0), (BF_CVT_I, MODIFIER, ".rz", 0),
        (BF_CVT_I, MODIFIER, ".rni", 1), (BF_CVT_I, MODIFIER, ".rmi", 1),
        (BF_CVT_I, MODIFIER, ".rpi", 1), (BF_CVT_I, MODIFIER, ".rzi", 1),
        (BF_CVT_ABSSAT, MODIFIER, ".sat", 1),
        (BF_CVT_ABSSAT, MODIFIER, ".abs", 2),
        (BF_CVT_ABSSAT, MODIFIER, ".ssat", 3),
    ], [
    (DST1, f_oreg([16,32]),   dest_oreg),
    (SRC1, f_reg([16,32]),    reg_1(SRC1)), 
    (SRC1, f_s([16,32]),      shared_1(SRC1)),
]),
# integer to float, scaling
("cvt", 2, wide_op(0xa3) + pred_out + pred_in + _always(".scale"), [
    (DST1, f_oreg([32]),     dest_oreg),
    (SRC1, (OP_TYPE_INT, _s.US, [32],   OP_SOURCE_REGISTER, OP_INDIRECTION_NONE),      reg_1(SRC1)), 
    (SRC1, i_s(_s.US, [32]),    shared_1(SRC1)),
]),

# logic ops
("and", 2, wide_op(0xd0) + pred_out + pred_in + psize_width2 + _logic(0),  std_operands_1_3(_s.B)),
("or",  2, wide_op(0xd0) + pred_out + pred_in + psize_width2 + _logic(1),  std_operands_1_3(_s.B)),
("xor", 2, wide_op(0xd0) + pred_out + pred_in + psize_width2 + _logic(2),  std_operands_1_3(_s.B)),
# 0  0x1 0001 a and b
# 1  0x7 0111 a or b
# 2  0x6 0110 a xor b
# 3  0x5 0101 b
# 4  0x4 0100 (not a) and b
# 5  0xd 1101 (not a) or b
# 6  0x9 1001 (not a) xor b
# 7  0x5 0101 b
# 8  0x2 0010 a and (not b)
# 9  0xb 1011 a or (not b)
# 10 0x9 1001 a xor (not b)
# 11 0xa 1010 not b
# 12 0x8 1000 (not a) and (not b) (nor)
# 13 0xe 1110 (not a) or (not b) (nand)
# 14 0x6 0110 (not a) xor (not b) 
# 15 0xa 1010 not b
# special case
("not", 2, wide_op(0xd0) + pred_out + pred_in + psize_width1 + subsubop(11), [
    (DST1, i_oreg(_s.B), dest_oreg),
    (SRC1, i_reg(_s.B),  reg_3(SRC1)), 
    (SRC1, i_s(_s.B),    const_3(SRC1)),
]),

# floating point operations (single operand)
("rcp",   2, wide_op(0x90) + pred_out + pred_in, std_operands_f1()),
# 0x91 invalid
("rsqrt", 2, wide_op(0x92) + pred_out + pred_in, std_operands_f1()),
("lg2",   2, wide_op(0x93) + pred_out + pred_in, std_operands_f1()),
("sin",   2, wide_op(0x94) + pred_out + pred_in, std_operands_f1()),
("cos",   2, wide_op(0x95) + pred_out + pred_in, std_operands_f1()),
("ex2",   2, wide_op(0x96) + pred_out + pred_in, std_operands_f1()),
# 0x97 invalid

# Add small delta value to a float, away from 0
# subsubop is 0 or 1
("delta",   2, wide_op(0xa7) + pred_out + pred_in + subsubop(1), std_operands_f1()),

# floating point operations (two operands)
# normal add
("add", 2,  wide_op(0xb0) + pred_out + pred_in + _rmf(BF_FLOP_ROP) + _flop_flip, std_operands_f1_4()),
# subtract with reversed operands
# XXX I'm not sure about this one, the alt bit might also invert result
#     seems alt bit does nothing at all, at least for the long instruction
# ("subr", 2, wide_op(0xb0) + pred_out + pred_in + bit(BF_ALT) + _rmf(BF_FLOP_ROP) + _flop_flip, std_operands_f1_4()),

("max", 2, wide_op(0xb4) + pred_out + pred_in,  std_operands_f1_3()),
("min", 2, wide_op(0xb5) + pred_out + pred_in,  std_operands_f1_3()),
# flop6 (pre.sin, pre.ex2)
("pre", 2, wide_op(0xb6) + pred_out + pred_in + [
        (BF_SUBSUBOP, MODIFIER, ".sin", 0),
        (BF_SUBSUBOP, MODIFIER, ".lg2", 1)
    ],  std_operands_f1()),
("mul", 2, wide_op(0xc0) + pred_out + pred_in + _rmf(BF_SUBSUB_ROP), std_operands_f1_3()),
# ????
#("unk80", 2, wide_op(0x80) + pred_out + pred_in + _rmf(BF_SUBSUB_ROP), std_operands_f1_3()),

# XXX this instruction is weird; rounding op overlaps with constant segment
("mad", 2, wide_op(0xe0) + pred_out + pred_in + _rmf(BF_MAD_ROP) + [(BF_FLOP_FLIP1, SRC1, FLIP), (BF_FLOP_FLIP2, SRC3, FLIP)], [
    (DST1, f_oreg(), dest_oreg),
    (SRC1, f_reg(),  reg_1(SRC1)), 
    (SRC1, f_s(),    shared_1(SRC1)),
    (SRC2, f_reg(),  reg_3(SRC2)), 
    (SRC2, f_c(),    const_3(SRC2)),
#    (SRC2, f_imm(),  imm_3(SRC2)),
    (SRC3, f_reg(),  reg_4(SRC3))
]),

# slct
# sad
# texture operations
# unk80
# atomic ops
# ----------------------------------------------------------------------------
# 64 bit immediate instructions
# these are like the 32 bit instructions
# from immediate (long) to a register
("mov", 2, imm_op(0x10) + psize_width1_s, [
    (DST1, i_reg(_s.X),  dest),   
    (SRC1, i_imm(_s.X),  imm_3w(SRC1)), 
]),
# imm_op(0x20): add/sub
("add", 2, imm_op(0x20) + psize_width1_s, std_operands_imm1_3(_s.X)),
("sub", 2, imm_op(0x20) + psize_width1_s + bit(BF_ALT), std_operands_imm1_3(_s.X)),
# imm_op(0x30): subr/addc
("subr", 2, imm_op(0x30) + psize_width1_s, std_operands_imm1_3(_s.X)),
# Uses carry flag of p0
("addc", 2, imm_op(0x30) + psize_width1_s + bit(BF_ALT), std_operands_imm1_3(_s.X)),
# imm_op(0x40): mul24
("mul24", 2, imm_op(0x40) + _always(".lo") + [
        (BF_ALT, SRC1, IS_32BIT), (BF_ALT, SRC2, IS_32BIT), # source type depends on ALT bit
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US),  reg_1l(SRC1)), 
    (SRC1, i_s(_s.US),    shared_1l(SRC1)),
    (SRC2, i_imm(_s.US),  imm_3w(SRC2)), 
    ]),
# imm_op(0x50): sad
("sad", 2, imm_op(0x50) + psize_width2_s, [
    (DST1, i_reg(_s.X),  dest), 
    (SRC1, i_reg(_s.X),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.X),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.X),    shared_1l(SRC2)),
    (SRC3, i_imm(_s.X),  imm_3w(SRC3)), 
    ]),
# imm_op(0x60): mac16 XXX
("mad24", 2, imm_op(0x60) + _always(".lo") + [
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED),
        (BF_ALT, SRC1, FLIP)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US, 32),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.US, 16),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.US, 16),    shared_1l(SRC2)),
    (SRC3, i_imm(_s.US, 16),  imm_3w(SRC3)), 
    ]),
# imm_op(0x70): ?
#   SRC1 and SRC2 are always 16 bit
#   BF_OPER1_H, IS_SIGNED
#   mad24, but b is flipped
#   bit(BF_ALT) flips c
#   XXX this must be used of b is flipped
("op70", 2, imm_op(0x70) + _always(".lo") + [
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED),
        (BF_ALT, SRC3, FLIP)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US, 32),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.US, 16),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.US, 16),    shared_1l(SRC2)),
    (SRC3, i_imm(_s.US, 16),  imm_3w(SRC3)), 
    ]),
# imm_op(0x90): rcp, does this make sense?
#("rcp", 1, imm_op(0x90) + [(BF_ALT, SRC1, FLIP)], std_operands_fl1()),
# imm_op(0xa0): cvt?
#   usage crashes kernel
("add", 2, imm_op(0xb0) + [(BF_ALT, SRC2, FLIP), (BF_OPER1_H, SRC1, FLIP)] + _always(".rn"), std_operands_immf1_3()),
# imm_op(0xc0): mulf
("mul", 2, imm_op(0xc0) + [(BF_ALT, SRC2, FLIP), (BF_OPER1_H, SRC1, FLIP)] + _always(".rn"), std_operands_immf1_3()),
# imm_op(0xd0): logic?
#   usage crashes kernel
("mad", 2, imm_op(0xe0) + [(BF_ALT, SRC1, FLIP), (BF_OPER1_H, SRC2, FLIP)] + _always(".rn"), [
    (DST1, f_reg(),  dest), 
    (SRC1, f_reg(),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, f_reg(),  reg_1l(SRC2)), 
    (SRC2, f_s(),    shared_1l(SRC2)),
    (SRC3, f_imm(),  imm_3w(SRC3)), 
    ]),

# ----------------------------------------------------------------------------
# half (32 bit) instructions
# only arguments 2 <- 1,3
# output only register
# no predication, no subsubop (but there is alt bit)
# BF_OPER3_CONST (always segment 0?)
# BF_OPER3
# offset registers 1-3 addressable
# BF_OPER2    arg2
# BF_OPER1_L  arg1
# BF_OPER1_SHARED_H  -> take from shared
# half_op(0x00): ?
# half_op(0x10): ld
("mov", 1, half_op(0x10) + psize_width1_s + halfflag, [
    (DST1, i_reg(_s.X),  dest),
    (SRC1, i_reg(_s.X),  reg_1l(SRC1)), 
    (SRC1, i_s(_s.X),    shared_1l(SRC1)),
]),
# half_op(0x20): add/sub
("add", 1, half_op(0x20) + psize_width1_s + halfflag, std_operands_l1_3(_s.X)),
("sub", 1, half_op(0x20) + psize_width1_s + bit(BF_ALT) + halfflag, std_operands_l1_3(_s.X)),
# half_op(0x30): subr/addc
("subr", 1, half_op(0x30) + psize_width1_s + halfflag, std_operands_l1_3(_s.X)),
# Uses carry flag of p0
("addc", 1, half_op(0x30) + psize_width1_s + bit(BF_ALT) + halfflag, std_operands_l1_3(_s.X)),
# half_op(0x40): mul24
("mul24", 1, half_op(0x40) + halfflag + _always(".lo") + [
        (BF_ALT, SRC1, IS_32BIT), (BF_ALT, SRC2, IS_32BIT), # source type depends on ALT bit
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US),  reg_1l(SRC1)), 
    (SRC1, i_s(_s.US),    shared_1l(SRC1)),
    (SRC2, i_reg(_s.US),  reg_3l(SRC2)), 
    (SRC2, i_c(_s.US),    const_3l(SRC2)),
    ]),
# half_op(0x50): sad
#   a = a + |b - c|
# alt bit seems to do nothing
("sad", 1, half_op(0x50) + psize_width2_s + halfflag, [
    (DST1, i_reg(_s.X),  dest), 
    (SRC1, i_reg(_s.X),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.X),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.X),    shared_1l(SRC2)),
    (SRC3, i_reg(_s.X),  reg_3l(SRC3)), 
    (SRC3, i_c(_s.X),    const_3l(SRC3)),
    ]),

#("sada", 1, half_op(0x50) + psize_width2_s + halfflag + bit(BF_ALT), std_operands_l1_3(_s.X)),
# half_op(0x60): mac16 XXX
#   a = a + (b * c)
#    bit(BF_ALT)      flips a
#   SRC1 and SRC2 are always 16 bit
#   DST1 is always 32 bit
#   BF_OPER1_H, IS_SIGNED
("mad24", 1, half_op(0x60) + halfflag + _always(".lo") + [
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED),
        (BF_ALT, SRC1, FLIP)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US, 32),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.US, 16),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.US, 16),    shared_1l(SRC2)),
    (SRC3, i_reg(_s.US, 16),  reg_3l(SRC3)), 
    (SRC3, i_c(_s.US, 16),    const_3l(SRC3)),
    ]),
# half_op(0x70): ?
#   SRC1 and SRC2 are always 16 bit
#   BF_OPER1_H, IS_SIGNED
#   mad24, but b is flipped
#   bit(BF_ALT) flips c
#   XXX this must be used of b is flipped
("op70", 1, half_op(0x70) + halfflag + _always(".lo") + [
        (BF_OPER1_H, DST1, IS_SIGNED), (BF_OPER1_H, SRC1, IS_SIGNED), (BF_OPER1_H, SRC2, IS_SIGNED),
        (BF_ALT, SRC3, FLIP)
    ], [
    (DST1, i_reg(_s.US, 32),  dest), # dest is always 32 bit
    (SRC1, i_reg(_s.US, 32),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, i_reg(_s.US, 16),  reg_1l(SRC2)), 
    (SRC2, i_s(_s.US, 16),    shared_1l(SRC2)),
    (SRC3, i_reg(_s.US, 16),  reg_3l(SRC3)), 
    (SRC3, i_c(_s.US, 16),    const_3l(SRC3)),
    ]),
# half_op(0x80): load from interpolator (faults)
# ("op80", 1, half_op(0x80) + halfflag + bit(BF_ALT), std_operands_l1_3(_s.X)),
# half_op(0x90): rcp
#   a = 1.0f/b
#   is a floating point instruction
#   bit(BF_ALT)       flips b
("rcp", 1, half_op(0x90) + halfflag + [(BF_ALT, SRC1, FLIP)] + _always(".rn"), std_operands_fl1()),
# half_op(0xa0): cvt?
#   usage crashes kernel
# ("cvt", 1, half_op(0xa0) + halfflag + bit(BF_ALT), std_operands_l1_3(_s.X)),
# half_op(0xb0): addf
#    a = b + c
#    bit(BF_OPER1_H)  flips b
#    bit(BF_ALT)      flips c
("add", 1, half_op(0xb0) + halfflag + [(BF_ALT, SRC2, FLIP), (BF_OPER1_H, SRC1, FLIP)] + _always(".rn"), std_operands_fl1_3()),
# half_op(0xc0): mulf
#    a = b * c
#    bit(BF_OPER1_H)  flips b
#    bit(BF_ALT)      flips c
("mul", 1, half_op(0xc0) + halfflag + [(BF_ALT, SRC2, FLIP), (BF_OPER1_H, SRC1, FLIP)] + _always(".rn"), std_operands_fl1_3()),
# half_op(0xd0): logic?
#   usage crashes kernel
# ("and", 1, half_op(0xd0) + halfflag, std_operands_l1_3(_s.X)), # l1_3f
# half_op(0xe0): madf (more like mac)
#    a = a + (b * c)
#    bit(BF_OPER1_H)  flips b
#    bit(BF_ALT)      flips a
("mad", 1, half_op(0xe0) + halfflag + [(BF_ALT, SRC1, FLIP), (BF_OPER1_H, SRC2, FLIP)] + _always(".rn"), [
    (DST1, f_reg(),  dest), 
    (SRC1, f_reg(),  [(BF_OPER2, SRC1, VALUE)]), # first source is same as dest
    (SRC2, f_reg(),  reg_1l(SRC2)), 
    (SRC2, f_s(),    shared_1l(SRC2)),
    (SRC3, f_reg(),  reg_3l(SRC3)), 
    (SRC3, f_c(),    const_3l(SRC3)),
    ]),
# half_op(0xf0): tex
# XXX todo
# ----------------------------------------------------------------------------
# flow control (64 bit) instructions
("kil",    2, s_op(0x00) + pred_in, []),
("bra",    2, s_op(0x10) + pred_in, oper_addr),
("call",   2, s_op(0x20) + pred_in, oper_addr),
("return", 2, s_op(0x30) + pred_in, []),
("breakaddr",2, s_op(0x40), oper_addr),
("break",  2, s_op(0x50) + pred_in, []),
("bar",    2, s_op(0x80) + [(BF_ALWAYS, MODIFIER, ".sync", 1), (BF_SYNC_OP, IMM, 3), (BF_SYNC_ARG, IMM, 0xFFF)], [
    (DST1, (OP_TYPE_INT, _s.X, 32, OP_SOURCE_IMMEDIATE, OP_INDIRECTION_NONE), [(BF_SYNC_BAR, DST1, VALUE)]),
]),
("trap",   2, s_op(0x90), []),
("join",   2, s_op(0xA0), oper_addr), 

]
