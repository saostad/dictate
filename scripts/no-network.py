#!/usr/bin/env python3
"""Run a local verification command with network syscalls blocked on Linux.

Usage: python3 scripts/no-network.py target/release/dictate benchmark ...
Requires libseccomp.so.2. Failure to install the filter aborts the command.
The filter survives exec and is inherited by child processes and threads.
"""

import ctypes
import errno
import os
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: no-network.py COMMAND [ARG ...]")
    if not sys.platform.startswith("linux"):
        raise SystemExit("no-network.py requires Linux and libseccomp.so.2")

    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint
    ]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None

    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise SystemExit("seccomp_init failed")
    try:
        for name in (b"socket", b"connect", b"sendto", b"sendmsg"):
            syscall = lib.seccomp_syscall_resolve_name(name)
            if syscall < 0:
                raise SystemExit(f"unknown syscall: {name.decode()}")
            result = lib.seccomp_rule_add(
                context, 0x00050000 | errno.EPERM, syscall, 0
            )  # SCMP_ACT_ERRNO(EPERM)
            if result != 0:
                raise SystemExit(f"seccomp_rule_add failed: {result}")
        result = lib.seccomp_load(context)
        if result != 0:
            raise SystemExit(f"seccomp_load failed: {result}")
    finally:
        lib.seccomp_release(context)

    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
