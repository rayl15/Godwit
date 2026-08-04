#!/usr/bin/env python3
"""Chat with a Godwit installation.

The Swift side has no tokeniser yet, so this stands in. Two things matter:

  o200k_harmony from tiktoken matches the model's 201,088-token vocabulary.

  GPT-OSS is instruction-tuned for the harmony conversation format. Fed raw
  text it produces one good continuation and then drifts, because bare prose is
  out of distribution for it. Wrapped properly it emits a structured reply with
  an analysis channel before the answer.

usage: python3 Scripts/chat.py <gwt-dir> "prompt" [n_tokens] [--raw]
"""
import subprocess
import sys

import tiktoken

model, prompt = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 32
raw = "--raw" in sys.argv

enc = tiktoken.get_encoding("o200k_harmony")
special = enc._special_tokens

if raw:
    ids = enc.encode(prompt)
else:
    ids = []
    for role, content in [("system", "You are a helpful assistant."), ("user", prompt)]:
        ids += ([special["<|start|>"]] + enc.encode(role) + [special["<|message|>"]]
                + enc.encode(content) + [special["<|end|>"]])
    ids += [special["<|start|>"]] + enc.encode("assistant")

print(f"prompt ({len(ids)} tokens): {prompt!r}", flush=True)

result = subprocess.run(
    [".build/release/godwit", "generate", "--model", model,
     "--tokens", ",".join(map(str, ids)), "--count", str(count)],
    capture_output=True, text=True)
if result.returncode != 0:
    sys.exit(result.stderr)

out = [int(x) for x in result.stdout.strip().split(",") if x]
print(f"\n{enc.decode(out)}")
