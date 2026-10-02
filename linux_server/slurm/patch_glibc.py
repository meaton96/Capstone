#!/usr/bin/env python3
"""Make UnityPlayer.so load on glibc 2.34 (RIT SPORC runs RHEL 9). Usage: python3 slurm/patch_glibc.py UnityPlayer.so

The Unity build needs glibc 2.35 for exactly one symbol, hypotf@GLIBC_2.35 (2.35 only swapped in a faster
implementation; hypotf@GLIBC_2.2.5 computes the same thing). This rebinds hypotf to the GLIBC_2.2.5 libm version and
marks the now-unused GLIBC_2.35 requirement weak, so the loader stops refusing the library. Edits the file in place;
safe to re-run (prints "already patched")."""
import struct, sys

VER_FLG_WEAK = 0x2


def main(path):
    data = bytearray(open(path, "rb").read())
    assert data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1, "expected a 64-bit little-endian ELF"
    e_shoff, = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    secs = []  # (type, offset, size, link, entsize)
    for i in range(e_shnum):
        _, sh_type, _, _, off, size, link, _, _, entsize = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        secs.append((sh_type, off, size, link, entsize))
    dynsym = next(s for s in secs if s[0] == 11)       # SHT_DYNSYM
    versym = next(s for s in secs if s[0] == 0x6fffffff)  # SHT_GNU_versym
    verneed = next(s for s in secs if s[0] == 0x6ffffffe)  # SHT_GNU_verneed
    dynstr_off = secs[dynsym[3]][1]
    vn_strtab = secs[verneed[3]][1]
    cstr = lambda base, o: data[base + o:data.index(b"\0", base + o)].decode()

    # Walk libm.so.6's version needs: find the indices of GLIBC_2.2.5 and GLIBC_2.35.
    idx, aux_235, off = {}, None, verneed[1]
    while True:
        _, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HHIII", data, off)
        if cstr(vn_strtab, vn_file) == "libm.so.6":
            a = off + vn_aux
            for _ in range(vn_cnt):
                _, flags, other, name, vna_next = struct.unpack_from("<IHHII", data, a)
                idx[cstr(vn_strtab, name)] = other
                if cstr(vn_strtab, name) == "GLIBC_2.35":
                    aux_235 = a
                a += vna_next
        if not vn_next:
            break
        off += vn_next
    if aux_235 is None:
        print(f"{path}: no libm GLIBC_2.35 requirement, nothing to do")
        return

    # Rebind every dynamic symbol that uses libm GLIBC_2.35 to libm GLIBC_2.2.5.
    names = []
    for i in range(dynsym[2] // dynsym[4]):
        vs_off = versym[1] + 2 * i
        v, = struct.unpack_from("<H", data, vs_off)
        if v & 0x7fff == idx["GLIBC_2.35"]:
            name, = struct.unpack_from("<I", data, dynsym[1] + i * dynsym[4])
            names.append(cstr(dynstr_off, name))
            struct.pack_into("<H", data, vs_off, (v & 0x8000) | idx["GLIBC_2.2.5"])
    unexpected = set(names) - {"hypotf"}
    assert not unexpected, f"only hypotf is known to be safe to rebind, also found {unexpected}"

    flags, = struct.unpack_from("<H", data, aux_235 + 4)
    if not names and flags & VER_FLG_WEAK:
        print(f"{path}: already patched")
        return
    struct.pack_into("<H", data, aux_235 + 4, flags | VER_FLG_WEAK)
    open(path, "wb").write(data)
    print(f"{path}: rebound {names} to GLIBC_2.2.5, GLIBC_2.35 requirement now weak")


if __name__ == "__main__":
    main(sys.argv[1])
