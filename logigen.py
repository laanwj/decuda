#!/bin/bash
# Generate logic operators according to the description in the G80 opengl 
# shader reference.
# http://developer.download.nvidia.com/opengl/specs/g80specs.pdf
# And compare to real output of a G80 card.
from Logic import *

tt = [[0]*16 for x in xrange(0,16)]

for x in xrange(0,16):
    zf = (x&ZF)!=0
    sf = (x&SF)!=0
    cf = (x&CF)!=0
    of = (x&OF)!=0
    
    tt[x][0] = False
    tt[x][1] = (sf and (not zf)) ^ of
    tt[x][2] = (not sf) and zf
    tt[x][3] = sf ^ (zf or of)
    tt[x][4] = ((not sf)^of) and (not zf)
    #tt[x][5] = sf or (not zf)
    tt[x][5] = not zf
    tt[x][6] = not (sf ^ of)
    tt[x][7] = (not sf) or (not zf)
    tt[x][8] = sf and zf
    tt[x][9] = not tt[x][6]
    tt[x][10] = not tt[x][5]
    tt[x][11] = not tt[x][4]
    tt[x][12] = not tt[x][3]
    tt[x][13] = not tt[x][2]
    tt[x][14] = not tt[x][1]
    tt[x][15] = not tt[x][0]


# 0 fl   0
# 1 lt   (SF && !ZF) ^ OF
# 2 eq   !SF && ZF
# 3 le   SF ^ (ZF || OF)
# 4 gt   (!SF ^ OF) && !ZF
# 5 ne   SF || !ZF
# 6 ge   !(SF ^ OF)
# 7 tr   1

def d(x,y):
    rv = str(int(tt[x][y]))
    if tt[x][y] != l[y][x]:
        rv += "*"
    else:
        rv += " "
    return rv

print "of  cf  sf  zf   fl  lt  eq  le  gt  ne  ge  leg nan ltu equ leu gtu neu geu tr "
for x in xrange(0,16):
    zf = (x&ZF)!=0
    sf = (x&SF)!=0
    cf = (x&CF)!=0
    of = (x&OF)!=0
    
    #i = [str(int(tt[x][y])) for y in xrange(0,16)]
    i = [d(x,y) for y in xrange(0,16)]
    str1 = "%i   %i   %i   %i    "%(of,cf,sf,zf)
    
    print str1+("  ".join(i))
     
