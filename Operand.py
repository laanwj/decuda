#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan <laanwj@gmail.com>, 2007

from cStringIO import StringIO
from Util import f2i

OP_TYPE_INT   = 0
OP_TYPE_FLOAT = 1
OP_TYPE_PRED  = 2 # 4 bit on G80
OP_TYPE_TEXTURE = 3

OP_SIGN_NONE     = 0 # b32 or f32
OP_SIGN_SIGNED   = 1 # s32
OP_SIGN_UNSIGNED = 2 # u32

OP_SOURCE_REGISTER = 1
OP_SOURCE_HALF_REGISTER = 2
OP_SOURCE_OFFSET_REGISTER = 3
OP_SOURCE_PRED_REGISTER = 4
OP_SOURCE_TEXTURE = 5
OP_SOURCE_IMMEDIATE = 6
OP_SOURCE_REGSET = 7 # Multiple registers
OP_SOURCE_OUTPUT_REGISTER = 8
OP_SOURCE_HALF_OUTPUT_REGISTER = 9

OP_INDIRECTION_NONE = 0
OP_INDIRECTION_SHARED = 1
OP_INDIRECTION_GLOBAL = 2
OP_INDIRECTION_LOCAL = 3
OP_INDIRECTION_CODE = 4
OP_INDIRECTION_INTERNAL = 5
# Constant segments 0..15
OP_INDIRECTION_CONST0 = 0x10
OP_INDIRECTION_CONST1 = 0x11
OP_INDIRECTION_CONST2 = 0x12
OP_INDIRECTION_CONST3 = 0x13
OP_INDIRECTION_CONST4 = 0x14
OP_INDIRECTION_CONST5 = 0x15
OP_INDIRECTION_CONST6 = 0x16
OP_INDIRECTION_CONST7 = 0x17
OP_INDIRECTION_CONST8 = 0x18
OP_INDIRECTION_CONST9 = 0x19
OP_INDIRECTION_CONST10 = 0x1A
OP_INDIRECTION_CONST11 = 0x1B
OP_INDIRECTION_CONST12 = 0x1C
OP_INDIRECTION_CONST13 = 0x1D
OP_INDIRECTION_CONST14 = 0x1E
OP_INDIRECTION_CONST15 = 0x1F
OP_OFFSET_NONE = None
OP_OFFSET_0 = 0  # Offset register 0
OP_OFFSET_1 = 1  # Offset register 1

_hilo = [".lo",".hi"]

_indir = {
OP_INDIRECTION_SHARED:"s", 
OP_INDIRECTION_GLOBAL:"g", 
OP_INDIRECTION_LOCAL:"l", 
OP_INDIRECTION_INTERNAL:"i",
OP_INDIRECTION_CONST0:"c0",
OP_INDIRECTION_CONST1:"c1",
OP_INDIRECTION_CONST2:"c2",
OP_INDIRECTION_CONST3:"c3",
OP_INDIRECTION_CONST4:"c4",
OP_INDIRECTION_CONST5:"c5",
OP_INDIRECTION_CONST6:"c6",
OP_INDIRECTION_CONST7:"c7",
OP_INDIRECTION_CONST8:"c8",
OP_INDIRECTION_CONST9:"c9",
OP_INDIRECTION_CONST10:"c10",
OP_INDIRECTION_CONST11:"c11",
OP_INDIRECTION_CONST12:"c12",
OP_INDIRECTION_CONST13:"c13",
OP_INDIRECTION_CONST14:"c14",
OP_INDIRECTION_CONST15:"c15",
}
_sign = ["b", "s", "u"]

# parameter space
_param_space = {
0x0: "%gridflags", # gridid
0x1: "%ntid.x",    # checked
0x2: "%ntid.y",
0x3: "%ntid.z",
0x4: "%nctaid.x",
0x5: "%nctaid.y",
0x6: "%ctaid.x",
0x7: "%ctaid.y",
#0x8: "%ctaid.z",  # extrapolated
#0x9: "%nctaid.y",  # ptx ISA
}

_ldgpu_ops = {
0:"%physid", 1:"%clock",
4:"%pm0", 5:"%pm1", 6:"%pm2", 7:"%pm3"
}
_ldgpu_ops_inv = dict([(y,x) for x,y in _ldgpu_ops.iteritems()])

_ofs_inc = ["+","+="]

class Operand(object):
    """Instruction operand (source or destination)"""
    type = None # int, float, predicate
    sign = None # signed, unsigned
    size = None # 64, 32, 16, 8
    source = None # register, immediate
    indirection = None # none, shared, global, local, const0, const1
    value = None # N
    offset = None # none, offset0, offset1
    label = None
    flip = False # -
    offset_inc = False
    invert = False # ~
    #multiplier = None # address multiplier for indirection
    
    def __init__(self, type=OP_TYPE_INT, sign=OP_SIGN_NONE, size=32, source=OP_SOURCE_IMMEDIATE, indirection=OP_INDIRECTION_NONE, value=0, offset=OP_OFFSET_NONE, flip=False, offset_inc=False, invert=False):
        self.type = type
        self.sign = sign
        self.size = size
        self.source = source
        self.indirection = indirection
        self.value = value
        self.offset = offset
        self.flip = flip
        self.offset_inc = offset_inc
        self.invert = invert
        #self.multiplier = multiplier

    def typestr(self):
        if self.source == OP_SOURCE_IMMEDIATE and self.indirection == OP_INDIRECTION_CODE and self.label != None:
            return ".label"
        if self.type == OP_TYPE_INT:
            return "."+_sign[self.sign]+str(self.size)
        elif self.type == OP_TYPE_FLOAT:
            return ".f"+str(self.size)
        else:
            return "" # don't show predicates as type

    def regstr(self, i):
        if i==-1:
            return "_"
        else:
            return "$r%i" % i
        
    def __repr__(self):
        rv = StringIO()
        value = self.value
        if self.flip:
            # sign bit
            rv.write("-")
        if self.source == OP_SOURCE_IMMEDIATE:
            if self.indirection in [OP_INDIRECTION_SHARED,OP_INDIRECTION_INTERNAL] or (self.indirection >= OP_INDIRECTION_CONST0 and self.indirection <= OP_INDIRECTION_CONST15):
                # We don't need that much precision for these limited memories
                width = "4"
            else:
                width = "8"
            times = ""
            #if self.multiplier:
            #    times = "%i*" % self.multiplier
            #else:
            #    times = ""
            if self.indirection == OP_INDIRECTION_NONE:
                if self.size == 8:
                    rv.write("0x%02x" % value)
                elif self.size == 16:
                    rv.write("0x%04x" % value)
                else: #self.size == 32:
                    rv.write("0x%08x" % value)
            elif self.indirection == OP_INDIRECTION_CODE:
                # look up labels
                if self.label != None:
                    rv.write(self.label)
                    #rv.write((" // (0x%06x)") % value)
                else:
                    rv.write(("0x%06x") % value)
            elif self.offset != None:
                rv.write(("%s[$ofs%i%s%s0x%0"+width+"x]") % (_indir[self.indirection], self.offset, _ofs_inc[self.offset_inc], times, value))
            elif self.indirection == OP_INDIRECTION_SHARED and self.size==16 and value in _param_space:
                rv.write(_param_space[value])
            elif self.indirection == OP_INDIRECTION_INTERNAL and value in _ldgpu_ops:
                rv.write(_ldgpu_ops[value])
            else:
                rv.write(("%s[%s0x%0"+width+"x]") % (_indir[self.indirection], times, value))
        elif self.source == OP_SOURCE_PRED_REGISTER:
            rv.write("$p%i" % value)
        elif self.source == OP_SOURCE_TEXTURE:
            rv.write("$tex%i" % value)
        elif self.source == OP_SOURCE_OUTPUT_REGISTER:
            rv.write("$o%i" % value)
        elif self.source == OP_SOURCE_REGISTER:
            if self.indirection == OP_INDIRECTION_NONE:
                rv.write("$r%i" % value)
            else:
                reg = "$r%i" % value
                if self.offset != None:
                    rv.write("%s[ofs%i%s%s]" % (_indir[self.indirection], self.offset, _ofs_inc[self.offset_inc], reg))
                else:
                    rv.write("%s[%s]" % (_indir[self.indirection], reg))
        elif self.source == OP_SOURCE_HALF_REGISTER:
            if self.indirection == OP_INDIRECTION_NONE:
                rv.write("$r%i%s" % (value/2, _hilo[value&1]))
            else:
                # Shared memory is addressed using 16 bits
                # XXX const too?
                reg = "$r%i%s" % (value/2, _hilo[value&1])
                if self.offset != None:
                    rv.write("%s[$ofs%i%s%s]" % (_indir[self.indirection], self.offset, _ofs_inc[self.offset_inc], reg))
                else:
                    rv.write("%s[%s]" % (_indir[self.indirection], reg))
        elif self.source == OP_SOURCE_OFFSET_REGISTER:
            rv.write("$ofs%i" % value)
        elif self.source == OP_SOURCE_REGSET:
            rv.write("{")
            rv.write(",".join([self.regstr(x) for x in self.value]))
            rv.write("}")
        else:
            rv.write([self.type,self.sign,self.size,self.source,self.indirection,self.value,self.offset].__repr__())
        return rv.getvalue()

    def parse(self, text, type):
        # parse type
        if type == "":
            self.type = OP_TYPE_PRED
        else:
            if type.startswith(".u"):
                self.type = OP_TYPE_INT
                self.sign = OP_SIGN_UNSIGNED
            elif type.startswith(".s"):
                self.type = OP_TYPE_INT
                self.sign = OP_SIGN_SIGNED
            elif type.startswith(".b"):
                self.type = OP_TYPE_INT
                self.sign = OP_SIGN_NONE
            elif type.startswith(".f"):
                self.type = OP_TYPE_FLOAT
                self.sign = OP_SIGN_NONE
            elif type == ".label":
                self.type = OP_TYPE_INT
                self.sign = OP_SIGN_NONE
                self.size = 32
                self.source = OP_SOURCE_IMMEDIATE
                self.indirection = OP_INDIRECTION_CODE
                self.value = 0 # value is yet unknown, substitute zero, will be filled in second pass
                self.label = text
                # we're finished here
                return
            else:
                raise ValueError("Invalid operand type %s" % type)
            try:
                self.size = int(type[2:])
            except ValueError:
                raise ValueError("Invalid operand type %s" % type)
        # parse value

        # Immediate value
        try:
            if text.endswith("f") and not text.startswith("0x"): # float imm
                self.value = f2i(float(text[0:-1]))
            else:
                self.value = int(text, 0)
            self.source = OP_SOURCE_IMMEDIATE
            return
        except ValueError:
            pass
        # Flip operand
        if text.startswith("-"):
            self.flip = True
            text = text[1:]
        # Invert operand
        if text.startswith("~"):
            self.invert = True
            text = text[1:]
        # Register
        sw = ['$tex', '$ofs', '$r', '$o', '$p']
        source_f = {0:OP_SOURCE_TEXTURE, 1:OP_SOURCE_OFFSET_REGISTER, 2:OP_SOURCE_REGISTER,      3:OP_SOURCE_OUTPUT_REGISTER, 4:OP_SOURCE_PRED_REGISTER}
        source_h = {                                                  2:OP_SOURCE_HALF_REGISTER, 3:OP_SOURCE_HALF_OUTPUT_REGISTER}
        for i,s in enumerate(sw):
            if text.startswith(s):
                break
        else:
            i = None
        if i != None:
            ls = len(s)
            try:
                dot = text.index(".")
            except ValueError:
                self.source = source_f[i]
                if self.source == OP_SOURCE_PRED_REGISTER:
                    self.size = 1
                    self.type = OP_TYPE_PRED
                self.value = int(text[ls:])
            else: 
                try:
                    self.source = source_h[i]
                except KeyError:
                    raise ValueError("%s cannot be divided" % text)
                if text[dot:] == ".lo":
                    self.value = int(text[ls:dot])*2
                elif text[dot:] == ".hi":
                    self.value = int(text[ls:dot])*2+1
                else:
                    raise ValueError("Invalid register operand %s" % text)
            return
        # bit bucket
        if text == "_":
            if self.size == 16:
                self.source = OP_SOURCE_HALF_OUTPUT_REGISTER
            else:
                self.source = OP_SOURCE_OUTPUT_REGISTER
            self.value = 0x7f
            return
        try:
            para = text.index("[")
        except ValueError:
            pass
        else: # indirection
            indir = text[0:para]
            for i, t in _indir.iteritems():
                if indir == t:
                    break
            else:
                raise ValueError("Invalid indirection %s" % indir)
            # i is now indirection type
            self.indirection = i
            # ] should be last character
            if not text.endswith("]"):
                raise ValueError("Indirection not terminated properly")
            value = text[para+1:-1]
            # expression within indirection can be
            # $ofs%i+0x1234 OP_SOURCE_IMMEDIATE  OP_OFFSET_x
            # $ofs%i+$r%i   OP_SOURCE_REGISTER   OP_OFFSET_x
            # 0x1234        OP_SOURCE_IMMEDIATE  OP_OFFSET_NONE
            # $r%i          OP_SOURCE_REGISTER   OP_OFFSET_NONE
            if value.startswith("$ofs"):
                try:
                    plus = value.index("+")
                    offreg = value[0:plus]
                    value = value[plus+1:]
                except ValueError:
                    offreg = value
                    value = ""
                # parse offset register
                try:
                    self.offset = int(offreg[4:])
                    self.offset_inc = False
                    if value.startswith("="):
                        self.offset_inc = True
                        value = value[1:]
                except ValueError:
                    raise ValueError("Invalid offset register specification %s" % offreg)
            else:
                self.offset = OP_OFFSET_NONE
            # parse value or register
            try:
                if value.startswith("$r"):
                    # register
                    self.source = OP_SOURCE_REGISTER
                    self.value = int(value[2:])
                else:
                    # immediate
                    self.source = OP_SOURCE_IMMEDIATE
                    if value == "": # $ofs without +
                        self.value = 0
                    else:
                        self.value = int(value, 0)
            except ValueError:
                raise ValueError("Invalid register or immediate specification %s" % value)
            return
        try:
            i = _ldgpu_ops_inv[text]
        except KeyError:
            pass
        else:
            self.source = OP_SOURCE_IMMEDIATE
            self.indirection = OP_INDIRECTION_INTERNAL
            self.value = i
            return 
        raise ValueError("Unknown operand %s" % text)
        
    def clone(self):
        from copy import copy
        return copy(self)
