#!/usr/bin/python
# Python script to generate logic operation table
from subprocess import Popen,PIPE
import sys

logic = []

for x in xrange(0, 32):
    text = """
    .entry my_kernel
    {
    .lmem 0
    .smem 32
    .reg 3
    .bar 0
    cvt.u32.u16 $r0, $r0.lo
    shl.u32 $r1, $r0, 0x00000002
    add.u32 $r1, s[0x0010], $r1
    #mov.u32 $r2, 0x7FFFFFFF
    #set.lt.u32 $p0|$r0, $r0, $r2
    #add.u32 $p0|$r0, $r0, $r2
    #mov.b32 $r0, $p0
    mov.b32 $p0, $r0
    mov.b32 $r0, 0
    mov.b32 $r2, 1
    @$p0.%i mov.b32 $r0, $r2
    mov.end.u32 g[$r1], $r0
    }
    """ % x
    f = open("insttest.asm","w")
    f.write(text)
    f.close()

    # Compile ptx
    sub = Popen(["../cudasm",'-o',"insttest.cubin","insttest.asm"], stdout=PIPE, stderr=PIPE)
    rv = sub.communicate()
    if sub.returncode:
        print "Compilation error"
        sys.exit(0)

    sub = Popen(["./exec"], stdout=PIPE, stderr=PIPE)
    rv = sub.communicate()
    if sub.returncode:
        print "Execution error"
        sys.exit(0)
    data = rv[0].strip()
    data = data.split(" ")
    data = [int(a, 16) for a in data]
    logic.append(data)
    
print logic
