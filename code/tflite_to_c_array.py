import sys
from pathlib import Path

def tflite_to_cc(tflite_path: str, out_cc_path: str, var_name: str):
    data = Path(tflite_path).read_bytes()

    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        lines.append(", ".join(f"0x{b:02x}" for b in chunk))

    body = ",\n    ".join(lines)

    cc = f"""#include <cstddef>
#include <cstdint>

alignas(16) const unsigned char {var_name}[] = {{
    {body}
}};

const size_t {var_name}_len = {len(data)};
"""

    Path(out_cc_path).write_text(cc, encoding="utf-8")
    print(f"Saved: {out_cc_path} ({len(data)} bytes)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tflite_to_c_array.py <in.tflite> <out.cc> <var_name>")
        raise SystemExit(1)

    tflite_to_cc(sys.argv[1], sys.argv[2], sys.argv[3])
