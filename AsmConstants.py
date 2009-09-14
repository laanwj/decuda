#!/usr/bin/python
# sm1_1 (G8x) disassembler (decuda)
# Wladimir J. van der Laan, 2007

# Some specific assembler constants, used in both the rule table as
# the assembler itself.

BF_ALWAYS       = None            # bitfield that is always 0xFFFFFFFF

# bitfield source -- src
IMM = 0       # immediate value
DST1 = 10     # first non-predicate destination operand
DST2 = 11     # second non-predicate destination operand
DST_LAST = 17
SRC1 = 20     # first source operand
SRC2 = 21     # second source operand
SRC3 = 22     # third source operand
SRC_LAST = 27
PRED_OUT = 28 # predicate out
PRED_IN = 29  # predicated instruction
MODIFIER = 30 # instruction modifier

# bitfield source property -- sub
(
VALUE,       # operand value
VALUE_ALIGN, # operand value as address, aligned to data type size
SHTYPE,      # shared memory operand type/signedness
OFFSET,      # offset register
OFFSET_INC,  # increment offset register
FLIP,        # operand is flipped (-)
INVERT,      # operand is inverted (~)
IS_OUTREG,   # source is output register
IS_SIGNED,   # type is signed
IS_32BIT,    # is 32 bit
CONSTSEG,    # constant segment
CC, 	     # condition code
PRESENT,     # output predicate is present?
GET_MSIZE,   # msize of operand
CVTI_TYPE,   # cvt(i) type
) = range(15)
