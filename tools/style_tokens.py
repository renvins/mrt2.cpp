"""Text prompt -> the 12 MusicCoCa style tokens, using Google's model.

A developer tool: mrt2.cpp can't turn text into style tokens yet (that
needs a port of MusicCoCa, plan step 13), so style presets are made here
once and built into the CLI.

Usage (from the mrt2 folder):
  MAGENTA_HOME=~/models .venv/bin/python tools/style_tokens.py "jazz piano trio" "techno"
"""
import sys

from magenta_rt import musiccoca


def main():
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
  model = musiccoca.MusicCoCa()
  for prompt in sys.argv[1:]:
    tokens = model.tokenize(model.embed(prompt)).tolist()[:12]
    print(f'{prompt!r}: {",".join(map(str, tokens))}')


if __name__ == '__main__':
  main()
