
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
    #mov.b32 $p0, $r0
    #mov.b32 $r0, 1
    #cvt.f32.u32 $r0, $r0
    #mov.b32 $r2, 0x7FFFFF
    #mov.b32 $r2, 0x800000
    #mov.b32 $r2, 2.0f
    #unk80.f32 $r0, $r0, $r0
    #cvt.f16.f32 $r2.lo, $r2
    #set.lt.u32.f32.f32 $r0, $r0, $r2
    
    #subr.f32 $r0, $r2, -$r0
    #add.f32 $r0, $r2, -$r0

    #cvt.s32.f32 $r0, $r0
    #mov.b32 $r2, 0xFFFFFFFC
# add          +a+b
# add.flip     +a-b
# subr         -a+b
# addc         +a+b+c
    #add.u32 $p3|$r0, $r0, $r2
    #mov.b32 $p0, $r0
    #mov.b32 $r2, 0x0
    #mov.b32 $p1, $r2
    #mov.b32 $p2, $r2
    #mov.b32 $p3, $r2

    #mov.b32 $r2, 0x7FFFFFFF
    # uses p0
    #@$p1.tr addc.u16 $r0.hi, $r0.hi, $r2.lo
    #sub.sat.s32 $r0, $r0, $r2
    #sub.s32 $r0, $r0, $r2

    #mov.b32 $r0, i[3]

    #mov.b32 $r2, 1
    #@$p0.31 mov.b32 $r0, $r2
    #mov.b32 $r2, 15
    #cvt.neg.s32.s16 $r0, $r0.lo
    #mov.b32 $ofs1, 0
    mov.b32 $ofs1, 0
    mov.b32 $r2, 0xFFFFFFFF

    mov.half.b32 $r1,$r1
    #add.b32 $r0,$r0,c1[$ofs1+$r2]
    add.half.b32 $r0,$r0,c1[$ofs1+0]

    #mov.b32 $r0, c0[$ofs1+=8]
    #mov.b32 $r0, c0[$ofs1+=4]
#s[$ofs1+12]

    mov.end.u32 g[$r1], $r0
    }
    