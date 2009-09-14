# Hi, I'm a comment
.entry my_kernel
{
    .lmem 0
    .reg 2
    .smem 32
    mov.b32 $r1, s[0x0010]
    mov.b32 $r0, s[0x0018]  // Another comment
    mov.u32 $r0, g[$r0]
    shl.u32 $r0, $r0, 0x00000002
    mov.end.u32 g[$r1], $r0
}
