#!/usr/bin/python
#
# Copyright 2010, Imran Haque
# ihaque AT cs DOT stanford DOT edu
# http://cs.stanford.edu/people/ihaque
# This code is placed in the public domain and bears no warranty whatsoever
#
# Uses objdump to dump contents of an nvcc-generated ELF file
# and parses this output to generate an old-format CUBIN file
#
# Alternatively, given the --nouveau option, parses ELF file
# and automatically passes each kernel off to one of the nouveau-project
# disassemblers nv50dis (Tesla) or nvc0dis (Fermi)
#
# Notable shortcomings: 
#    kernel .info sections are not parsed (presumably contains memory/reg info)
#    no attempt is made to preserve names of constant sections
#    does not parse size or proper name/offset of constant sections
#    emits initializer data for file-constant variables

import sys
import socket
from struct import unpack
from subprocess import Popen,PIPE
from StringIO import StringIO
import string

def ishexdigit(char):
    return (char in string.hexdigits)

class cubin:
    def __init__(self,codelevel="sm_10"):
        self.kernels = []
        self.consts = []
        self.info = []
        self.arch = codelevel
        self.abiversion = "1"
        self.modname = "cubin"
    def output(self,f):
        f.write("architecture {%s}\n"%self.arch)
        f.write("abiversion   {%s}\n"%self.abiversion)
        f.write("modname      {%s}\n"%self.modname)
        for c in self.consts:
            c.output(f)
        for k in self.kernels:
            k.output(f)

class constant:
    def __init__(self,name,data=None,depth=0,inkernel=False):
        self.name = name
        self.segname = "const"
        self.segnum = 0
        self.offset = 0
        self.bytes = 0
        self.depth = depth
        self.mem = None
        self.inkernel = inkernel
        if data is not None:
            self.mem = mem(data,self.depth+1)
    def output(self,f):
        prefix= "\t"*(self.depth+1)
        shortprefix= "\t"*(self.depth)
        if self.inkernel:
            f.write("%sconst {\n"%shortprefix)
        else:
            f.write("%sconsts {\n"%shortprefix)
            if self.name is not None:
                f.write("%s\t\tname    = %s\n"%(prefix,self.name))
        f.write("%s\t\tsegname = %s\n"%(prefix,self.segname))
        f.write("%s\t\tsegnum  = %d\n"%(prefix,self.segnum))
        f.write("%s\t\toffset  = %d\n"%(prefix,self.offset))
        f.write("%s\t\tbytes   = %d\n"%(prefix,self.bytes))
        if self.mem is not None:
            f.write("%smem {\n"%prefix)
            self.mem.output(f)
            f.write("%s}\n"%prefix)
        f.write("%s}\n"%shortprefix)

class kernel:
    def __init__(self,name,buf_text=None,buf_const=None,buf_info=None):
        self.name = name
        self.lmem = 0
        self.smem = 0
        self.reg = 0
        self.bar = 0
        self.offset = 0
        self.bytes = 0
        self.consts = []
        self.code = None
        if buf_text is not None:
            self.code = mem(buf_text)
        if buf_const is not None:
            self.consts = [constant("%s0"%self.name,buf_const,1,True)]
        if buf_info is not None:
            self.info = mem(buf_info,1)
    def output(self,f):
        f.write("code {\n")
        f.write("\tname = %s\n"%self.name)
        f.write("\tlmem = %d\n"%self.lmem)
        f.write("\tsmem = %d\n"%self.smem)
        f.write("\treg  = %d\n"%self.reg )
        f.write("\tbar  = %d\n"%self.bar )
        for c in self.consts:
            c.output(f)
        f.write("\tbincode {\n")
        self.code.output(f)
        f.write("\t}\n")
        f.write("}\n")
    def hex_output(self,f):
        self.code.hex_output(f)
        return
        

class mem:
    def __init__(self,data,depth=0,little_endian=True):
        self.le = little_endian
        self.data = []
        if data is not None:
            self.append(data)
        self.tabs = "\t"*(depth+2)
        return
    def append(self,data):
        for i in data:
            #print i
            x = int(i,16)
            if self.le:
                # This sort of assumes that the file was built on the same arch where it's being disasm'd
                x = socket.htonl(x)
            self.data.append(x)
        return
    def output(self,f):
        for i in range(0,len(self.data),4):
            datastrs = ["0x%08x"%x for x in self.data[i:i+4]]
            f.write("%s%s\n"%(self.tabs," ".join(datastrs)))
        return
    def hex_output(self,f):
        f.write("%s\n"%" ,".join(["0x%08x"%x for x in self.data]))
        return
   


def parse_objdump(objdump,codelevel,archlevel):
    # Parser states
    START        = 0
    INKERNEL     = 1
    INFILECONST  = 2
    INKCONST     = 3
    INKINFO      = 4
    INFINFO      = 5

    lines = [x.strip() for x in objdump.split("\n")]
    format = lines[1].split("-")[-1]
    little_endian = True
    if format != "little":
        little_endian = False
        print "Warning, found a non-little-endian file"
    buffer_text = []
    buffer_const = []
    buffer_info = []
    state = START
    kname = None
    const_id = 0
    file = cubin(codelevel)
    for line in lines[2:]:
        #print "%%:",line
        if len(line.strip()) == 0:
            continue
        if   state == START: #{{{
            if line.startswith("Contents of section .text"):
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif len(line) == 0:
                continue
            else:
                raise ValueError("Got unexpected line in state START\n%s"%line)
        #}}}
        elif state == INKERNEL: #{{{
            if line.startswith("Contents of section .text"):
                # Handle new kernel
                # Store old kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif line.startswith("Contents of section .nv.constant"):
                # Handle const data
                if line.endswith("%s:"%kname):
                    state = INKCONST
                    continue
                else:
                    # This is a file constant section
                    # Store old kernel
                    state = INFILECONST
                    file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                    kname = None
                    buffer_text = []
                    buffer_const = []
                    buffer_info = []
                    continue
            elif line.startswith("Contents of section .nv.info:"):
                # Entering a file info section, dump existing kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                kname = None
                state = INFINFO
            elif line.startswith("Contents of section .nv.info"):
                # Handle kernel info
                state = INKINFO
                continue
            elif ishexdigit(line[0]):
                # Handle new line of kernel binary
                # Take hex dump from objdump, remove address and ASCIIization
                buffer_text.extend(line[6:42].split())
            else:
                raise ValueError("Got unexpected line in state INKERNEL\n%s"%line)
        #}}}
        elif state == INFILECONST: #{{{
            if line.startswith("Contents of section .text"):
                # Handle new kernel
                # Store old kernel
                file.consts.append(constant("constant%d"%const_id,buffer_const))
                buffer_const = []
                const_id = 0
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif line.startswith("Contents of section .nv.constant"):
                # Handle const data
                # we should not be in a kernel here
                # TODO make sure not in kernel
                file.consts.append(constant("constant%d"%const_id,buffer_const))
                buffer_const = []
                const_id = 0
                state = INFILECONST
                continue
            elif line.startswith("Contents of section .nv.info:"):
                # Entering a file info section, dump existing section
                file.consts.append(constant("constant%d"%const_id,buffer_const))
                buffer_const = []
                const_id = 0
                kname = None
                state = INFINFO
            elif line.startswith("Contents of section .nv.info"):
                # Handle kernel info
                raise ValueError("Error, got into an .nv.info section from a file const section")
            elif ishexdigit(line[0]):
                # Handle new line of kernel binary
                # Take hex dump from objdump, remove address and ASCIIization
                buffer_const.extend(line[6:42].split())
            else:
                raise ValueError("Got unexpected line in state INFILECONST\n%s"%line)
        #}}}
        elif state == INKCONST: #{{{
            if line.startswith("Contents of section .text"):
                # Handle new kernel
                # Store old kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                const_id = 0
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif line.startswith("Contents of section .nv.constant"):
                # Handle const data
                if line.endswith("%s:"%kname):
                    #raise ValueError("Can't deal yet with kernels with multiple const sections")
                    pass
                else:
                    # This is a file constant section
                    # Store old kernel
                    state = INFILECONST
                    file.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                    kname = None
                    buffer_text = []
                    buffer_const = []
                    buffer_info = []
                    const_id = 0
                    continue
            elif line.startswith("Contents of section .nv.info:"):
                # Entering a file info section, dump existing kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                kname = None
                state = INFINFO
            elif line.startswith("Contents of section .nv.info"):
                # Handle kernel info
                state = INKINFO
                continue
            elif ishexdigit(line[0]):
                # Handle new line of kernel binary
                # Take hex dump from objdump, remove address and ASCIIization
                buffer_const.extend(line[6:42].split())
            else:
                raise ValueError("Got unexpected line in state INKCONST\n%s"%line)
        #}}}
        elif state == INKINFO: #{{{
            if line.startswith("Contents of section .text"):
                # Handle new kernel
                # Store old kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                const_id = 0
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif line.startswith("Contents of section .nv.constant"):
                # Handle const data
                if line.endswith("%s:"%kname):
                    state = INKCONST
                    continue
                else:
                    # This is a file constant section
                    # Store old kernel
                    state = INFILECONST
                    file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                    kname = None
                    buffer_text = []
                    buffer_const = []
                    buffer_info = []
                    continue
            elif line.startswith("Contents of section .nv.info:"):
                # Entering a file info section, dump existing kernel
                file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))
                buffer_text = []
                buffer_const = []
                buffer_info = []
                kname = None
                state = INFINFO
            elif line.startswith("Contents of section .nv.info"):
                # Handle kernel info
                raise ValueError("Can't handle kernels with multiple INFO sections")
                state = INKINFO
                continue
            elif ishexdigit(line[0]):
                # Handle new line of kernel binary
                # Take hex dump from objdump, remove address and ASCIIization
                buffer_info.extend(line[6:42].split())
            else:
                raise ValueError("Got unexpected line in state INKINFO\n%s"%line)
        #}}}
        elif state == INFINFO: #{{{
            if line.startswith("Contents of section .text"):
                # Handle new kernel
                # Store old kernel
                file.info.append(mem(buffer_info))
                buffer_info = []
                kname = line.split(".")[-1].rstrip(":")
                state = INKERNEL
                continue
            elif line.startswith("Contents of section .nv.constant"):
                # Handle const data
                # we should not be in a kernel here
                # TODO make sure not in kernel
                file.info.append(mem(buffer_info))
                buffer_info = []
                state = INFILECONST
                continue
            elif line.startswith("Contents of section .nv.info:"):
                # Entering a file info section, dump existing section
                file.info.append(mem(buffer_info))
                buffer_info = []
                state = INFINFO
            elif line.startswith("Contents of section .nv.info"):
                # Handle kernel info
                raise ValueError("Error, got into an .nv.info.KERNEL section from a file info section")
            elif ishexdigit(line[0]):
                # Handle new line of kernel binary
                # Take hex dump from objdump, remove address and ASCIIization
                buffer_info.extend(line[6:42].split())
            else:
                raise ValueError("Got unexpected line in state INFINFO\n%s"%line)
        #}}}
        else:
            raise ValueError("wtf bad state")
    if kname is None:
        if len(buffer_const) > 0:
            file.consts.append(constant("constant%d"%const_id,buffer_const))
    else:
        file.kernels.append(kernel(kname,buffer_text,buffer_const,buffer_info))

    return file

def parseHeader(elfname):
    # Parse compute level from elf header
    elf = open(elfname,"rb")
    elf.seek(4)
    elflevel = unpack('B',elf.read(1))[0]
    if elflevel == 1:
        # ELF32 has flags at 36 bytes in
        elf.seek(36)
    elif elflevel == 2:
        # ELF64 has flags at 48 bytes in
        elf.seek(48)
    else:
        raise ValueError("Invalid elflevel %x"%elflevel)
    flags = unpack("I",elf.read(4))[0]
    elf.close()
    #print "0x%08x"%flags
    codeint = flags & 0xFFFF
    archint = (flags & 0xFFFF0000) >> 16
    code = "sm_%d"%codeint
    arch = "sm_%d"%archint
    #print "code",code, "arch",arch
    return (code,codeint,arch,archint)

if len(sys.argv) < 2:
    print "Usage: elfToCubin [--nouveau] [input file]"
    sys.exit(1)

nouveau = False
elfname = None
if sys.argv[1] == '--nouveau':
    nouveau = True
    if len(sys.argv) < 3:
        print "Usage: elfToCubin [--nouveau] [input file]"
        sys.exit(1)
    elfname = sys.argv[2]
else:
    elfname = sys.argv[1]

(code,code_int,arch,arch_int) = parseHeader(elfname)

output = "".join(Popen(["objdump", "-s", elfname], stdout=PIPE).communicate()[0])
cubin = parse_objdump(output,code,arch)
if not nouveau:
    cubin.output(sys.stdout)
else:
    disassembler = None
    if code_int < 20:
        disassembler = "nv50dis"
    else:
        disassembler = "nvc0dis"
    for kernel in cubin.kernels:
        print "--> Disassembling kernel ",kernel.name,"with",disassembler
        disasm = Popen(disassembler,stdin=PIPE,stdout=PIPE)
        hex = StringIO()
        kernel.hex_output(hex)
        output = "".join(disasm.communicate(hex.getvalue())[0])
        hex.close()
        print output
        print
