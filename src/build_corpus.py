"""Harvest real Python from the stdlib into one character-level corpus.

Character level, deliberately. A ~100-symbol vocabulary means Network B can
read the embedding table directly - every row is one character you can name.
Byte-pair tokens would be faster to train and far harder to interpret.
"""
import os, sys, glob, json, sysconfig, random

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data'
SKIP = ("/test/", "/tests/", "/idlelib/", "/lib2to3/", "site-packages", "/encodings/")

def main():
    stdlib = sysconfig.get_paths()["stdlib"]
    files = [f for f in glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True)
             if not any(s in f for s in SKIP)]
    files.sort()
    random.Random(0).shuffle(files)

    chunks, kept = [], 0
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(src) < 200:
            continue
        # drop files that are mostly non-ASCII (unicode data tables, etc.)
        if sum(ord(c) > 127 for c in src) / len(src) > 0.01:
            continue
        chunks.append(src)
        kept += 1

    text = "\n\n".join(chunks)
    text = "".join(c for c in text if c == "\n" or c == "\t" or 32 <= ord(c) < 127)

    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}

    with open(os.path.join(OUT, "corpus.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump({"vocab": vocab, "stoi": stoi, "n_chars": len(text)}, fh)

    print(f"  files kept   : {kept:,} / {len(files):,}")
    print(f"  characters   : {len(text):,}")
    print(f"  vocab size   : {len(vocab)}")
    print(f"  vocab        : {''.join(vocab).replace(chr(10),'\\n').replace(chr(9),'\\t')}")
    print(f"\n  sample:\n" + "\n".join("    " + l for l in text[3000:3400].split("\n")))

if __name__ == "__main__":
    main()
