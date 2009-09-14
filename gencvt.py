#!/usr/bin/python
from AutoComp import PTX,CompilationError
import sys

ptx = """
.version 1.0
.target compute_10, map_f64_to_f32

.entry bra_test
{
.reg .%(src)s $r1;
.reg .u64 $rd1,$rd2;
.reg .%(dst)s $f1;
.param .u64 __cudaparm_data;
.param .u64 __cudaparm_data2;

ld.param.u64 	$rd1, [__cudaparm_data2];
ld.global.%(src)s 	$r1, [$rd1+0]; 
cvt%(round)s.%(dst)s.%(src)s 	$f1, $r1;
ld.param.u64 	$rd2, [__cudaparm_data];
st.global.%(dst)s 	[$rd2+0], $f1;

exit;
}
"""


types = ["u8","u16","u32","s8","s16","s32","f32"] #"f16", "f64","u64","s64"
modes_fp = [".rn", ".rz", ".rm", ".rp"]
modes_int = [".rni", ".rzi", ".rmi", ".rpi"]

#vars = {"src":"s32", "dst":"f32", "round":".rn"}
#x = PTX(ptx % vars)
#disa = x.bin.kernels[0].disassemble()

for src in types:
    for dst in types:
        modes = [""]
        if dst[0]=="f":
            modes = modes_fp
        else:
            modes = modes_int
        for mode in modes:
            print src,dst,mode
            vars = {"src":src, "dst":dst, "round":mode}
            print src, dst, mode
            try:
                x = PTX(ptx % vars)
                disa = x.bin.kernels[0].disassemble(sys.stdout)
                #lines = disa.split("\n")
                #if len(lines)>5:
                #    print lines[3]
            except CompilationError,e:
                print e.message
            print "====================="
