import re
import subprocess
import sys

KERNEL = re.compile(r"detail::baseline::(BankKernel::|TrackingKernel::|processBank|processTracking|sanitizeTracking)")
ENTRY = re.compile(r"detail::baseline::(processBank|processTracking<\(fckfft::\w+::TrackingRule\)\d>|sanitizeTracking)\(")
CALL = re.compile(r"^\s*[0-9a-f]+:\s+blx?\s+[0-9a-f]+ <(.+)>$")
ALLOWED_CALLS = ("memset", "memcpy")


def main(objdump, elf):
    listing = subprocess.run([objdump, "-d", "-C", "--no-show-raw-insn", elf], check=True,
                             capture_output=True, text=True).stdout
    function = None
    entries = set()
    errors = []
    for line in listing.splitlines():
        header = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if header:
            function = header.group(1) if KERNEL.search(header.group(1)) else None
            if function:
                entry = ENTRY.search(function)
                if entry:
                    entries.add(entry.group(1))
            continue
        if function is None:
            continue
        if ".f64" in line:
            errors.append(f"double precision in {function}: {line.strip()}")
        call = CALL.match(line)
        if call and not KERNEL.search(call.group(1)) and not call.group(1).startswith(ALLOWED_CALLS):
            errors.append(f"call in {function}: {line.strip()}")
    if len(entries) != 6:
        errors.append(f"found {len(entries)} of 6 kernel entry points: {sorted(entries)}")
    for error in errors:
        print(error)
    print(f"{len(entries)} kernel entry points, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
