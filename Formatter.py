#!/usr/bin/python
# sm1_1 (G80) disassembler (decuda)
# Wladimir J. van der Laan, 2007

# Base formatter
class Formatter:
    def kernel(self, rv, num, name):
        rv.write("// Disassembling %s\n" % (name))

    def address(self, rv, address):
        rv.write("%06x: " % (address))
    def bincode(self, rv, d):
        rv.write("%-17s " % d)
    def newline(self, rv):
        rv.write("\n")
        
    def pred(self, rv, s):
        rv.write(s+" ")
    def base(self, rv, s):
        rv.write(s)
    def modifier(self, rv, s):
        rv.write(s)
    def types(self, rv, s):
        rv.write(s+" ")
    def dest_operands(self, rv, s):
        rv.write(s)
    def src_operands(self, rv, s):
        rv.write(s)
    def label(self, rv, s):
        rv.write(s+": ")
    def warning(self, rv, w):
        rv.write("// ("+w+")")

    def const_hdr(self, rv, name, num, offset, size):
        rv.write("// segment: %s (%01x:%04x)\n" % (name,num,offset))
    def const_data(self, rv, mem):
        #rv.write("%s\n" % (mem))
        ofs = 0
        while ofs < len(mem):
            rv.write("%04x: " % (ofs*4))
            try:
                for x in xrange(0, 4):
                    rv.write("%08x " % mem[ofs+x])
            except IndexError:
                pass # expected at end of data
            rv.write("\n")
            ofs += 4


class AnsiFormatter(Formatter):
    def kernel(self, rv, num, name):
        rv.write("\x1b[0m// Disassembling \x1b[1;33m%s\x1b[0m\x1b[0m\n" % (name))


    def address(self, rv, address):
        rv.write("\x1b[1;30m%06x\x1b[0m: " % (address))
    def bincode(self, rv, d):
        rv.write("\x1b[0m%-17s\x1b[0m " % d)
    def newline(self, rv):
        rv.write("\x1b[0m\n")
        
    def pred(self, rv, s):
        rv.write("\x1b[0;32m"+s+"\x1b[0m ")
    def base(self, rv, s):
        rv.write("\x1b[0;34m"+s)
    def modifier(self, rv, s):
        rv.write(s)
    def types(self, rv, s):
        rv.write(s+"\x1b[0m ")
    def dest_operands(self, rv, s):
        rv.write("\x1b[1;34m"+s)
    def src_operands(self, rv, s):
        rv.write(s)
    def label(self, rv, s):
        rv.write("\x1b[33m"+s+":\x1b[0m ")

    def const_hdr(self, rv, name, num, offset, size):
        rv.write("// segment: \x1b[1;33m%s\x1b[0m (%01x:%04x)\n" % (name,num,offset))
    def const_data(self, rv, mem):
        #rv.write("%s\n" % (mem))
        ofs = 0
        while ofs < len(mem):
            rv.write("\x1b[1;30m%04x\x1b[0m: " % (ofs*4))
            try:
                for x in xrange(0, 4):
                    rv.write("%08x " % mem[ofs+x])
            except IndexError:
                pass # expected at end of data
            rv.write("\n")
            ofs += 4
    def warning(self, rv, w):
        rv.write(" \x1b[0;31m("+w+")\x1b[0m")
        
        
class AnsiFormatter2(Formatter):
    def address(self, rv, address):
        rv.write("\x1b[1;30m%06x\x1b[0m: " % (address))
    def bincode(self, rv, d):
        rv.write("\x1b[0m%-17s\x1b[0m " % d)
    def newline(self, rv):
        rv.write("\x1b[0m\n")
        
    def pred(self, rv, s):
        rv.write("\x1b[0;32m"+s+"\x1b[0m ")
    def base(self, rv, s):
        rv.write("\x1b[1;36m"+s)
    def modifier(self, rv, s):
        rv.write(s)
    def types(self, rv, s):
        rv.write(s+"\x1b[0m ")
    def dest_operands(self, rv, s):
        rv.write("\x1b[1;34m"+s)
    def src_operands(self, rv, s):
        rv.write(s)
    def label(self, rv, s):
        rv.write("\x1b[34m"+s+":\x1b[0m ")

class FileOutFormatter(Formatter):
    def address(self, rv, address):
        pass
    def warning(self, rv, w):
        rv.write("// ("+w+")")
    def bincode(self, rv, d):
        pass
    def const_hdr(self, rv, name, num, offset, size):
        rv.write("#.constseg %01x:0x%04x %s\n" % (num,offset,name))
        rv.write("#{\n")
    def const_data(self, rv, mem):
        #rv.write("%s\n" % (mem))
        ofs = 0
        while ofs < len(mem):
            rv.write("#d.u32 ")
            rv.write(", ".join(("0x%08x" % x) for x in mem[ofs:ofs+4]))
            rv.write(" // %04x\n" % (ofs*4))
            ofs += 4
        rv.write("#}\n")
