#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007
from Operand import *

# sm1_1 atomic operations
atomic_ops = [
"iadd","exch","cas","fadd","inc","dec","imax","imin","fmax",
"fmin","and","or","xor","????","????","????"
]

# memory operand sizes
#msize = [".u8",".s8",".u16",".s16",".64",".128",".u32",".s32"]

# logic operations (set)
logic_ops = {
0:".fl",1:".lt",2:".eq",3:".le",4:".gt",5:".ne",6:".ge",7:".tr"
}

# known http://developer.download.nvidia.com/opengl/specs/g80specs.pdf, 2.X.4.3, page 347:
# *u variants are inclusive NaN (see ptx guide, 7.3.1, page 33)
condition_codes = {
0:"fl",
1:"lt",
2:"eq",
3:"le",
4:"gt",
5:"ne",
6:"ge",
7:"leg",
8:"nan",
9:"ltu",  # (like LT, but allows NaN)
10:"equ", # (like EQ, but allows NaN)
11:"leu", # (like LE, but allows NaN)
12:"gtu", # (like GT, but allows NaN)
13:"neu", # (like NE, but allows NaN)
14:"geu", # (like GE, but allows NaN)
15:"tr",
16:"of", # overflow flag
17:"cf", # carry flag
18:"ab",
19:"sf", # sign flag
20:"20",
21:"21",
22:"22",
23:"23",
24:"24",
25:"25",
26:"26",
27:"27",
28:"nsf",
29:"ble",
30:"ncf",
31:"nof"
}
condition_codes_rev = dict([(y,x) for x,y in condition_codes.iteritems()])
condition_codes_rev["nzf"] = 5
condition_codes_rev["zf"] = 10

# op d subop 0 operations
d0_ops = {
0:"and",1:"or",2:"xor",
4:"norn",5:"nandn",6:"nxor",
8:"andn",9:"orn",10:"nxor",
11:"not"
}

cvti_types = {
0x0:(OP_SIGN_UNSIGNED,16,OP_TYPE_INT),  # .u16
0x1:(OP_SIGN_UNSIGNED,32,OP_TYPE_INT),  # .u32
0x2:(OP_SIGN_UNSIGNED,8,OP_TYPE_INT),   # .u8
0x4:(OP_SIGN_SIGNED,16,OP_TYPE_INT),    # .s16
0x5:(OP_SIGN_SIGNED,32,OP_TYPE_INT),    # .s32
0x6:(OP_SIGN_SIGNED,8,OP_TYPE_INT)      # .s8
}

sharedmem_types =  [# sign,size
(OP_SIGN_NONE, 8, OP_TYPE_INT),
(OP_SIGN_UNSIGNED, 16, OP_TYPE_INT),
(OP_SIGN_SIGNED, 16, OP_TYPE_INT),
(OP_SIGN_NONE, 32, OP_TYPE_INT)
]

# Rounding ops
cvt_rops = [".rn",".rm",".rp",".rz"]

abssat_ops = ["", ".sat", ".abs", ".ssat"]

