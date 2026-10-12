"""Dump reference intermediates from Google's JAX MRT2 into a .safetensors file.

This is a developer tool, never a runtime dependency. It runs Google's own
model once, records what every stage produces for each frame, and saves it
in the same format as the weights, so the C++ tests can read it with the
Weights class.

Settings that make the output comparable with C++:
  * float32 compute (Google's default is bfloat16, which would blur the
    comparison with our float32 C++),
  * greedy sampling (temperature 0): no random numbers, so it is
    deterministic and reproducible bit for bit.

Usage (from the mrt2 folder):
  MAGENTA_HOME=~/models JAX_PLATFORMS=cpu .venv/bin/python tools/dump_reference.py \
      --out ~/models/magenta-rt-v2/ref/ref_small.safetensors
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from magenta_rt.jax import model, system

# "jazz piano trio", tokenized by Google's MusicCoCa (see plan section 2).
JAZZ_PIANO_TRIO = [555, 402, 72, 424, 482, 932, 954, 153, 33, 374, 519, 779]
NUM_RESERVED = 6       # output token id = 6 + level * 1024 + code
CODEBOOK = 1024
LEVELS = 12


def calls(inter, path, method="step"):
  """All captured return values of `method` on the module at `path`."""
  d = inter
  for part in path.split("/"):
    d = d[part]
  return d[method]


def seq_values(call):
  """The array inside a captured (Sequence, state[, emits]) return value."""
  out = call[0] if isinstance(call, tuple) else call
  return np.asarray(out.values)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--frames", type=int, default=50)
  ap.add_argument("--codec-frames", type=int, default=10,
                  help="frames for which the (large) codec blocks are saved")
  args = ap.parse_args()

  # float32 everywhere: must be set before the model is built.
  model.MagentaRT2ModelBase.compute_dtype = jnp.float32
  mrt = system.MagentaRT2System(size="mrt2_small", temperature=0.0)

  block, constants = mrt._build_conditioning(
      {"mulan_tokens_25hz": JAZZ_PIANO_TRIO}, None, 0.0, None)
  state = mrt._jit_init_state(mrt._params, {})
  rngs = {"params": jax.random.PRNGKey(42), "random": jax.random.PRNGKey(0)}

  # One streaming step that also returns every sub-module's step output
  # (capture_intermediates) plus the values the model sow()s by name.
  @jax.jit
  def step(params, x, constants, state):
    (out, new_state, _), v = mrt._sampler.apply(
        params, x=x, state=state, constants=constants, training=False,
        rngs=rngs, method=mrt._sampler.step_with_emits,
        capture_intermediates=lambda m, meth: meth in ("step", "step_with_emits"),
        mutable=["intermediates"])
    return out, new_state, v["intermediates"]

  dec = "depthformer/decoder"
  ss = "soundstream/decoder"
  rec = {}  # name -> list of per-frame arrays

  def put(name, value):
    rec.setdefault(name, []).append(np.asarray(value, dtype=value.dtype))

  t0 = time.time()
  for f in range(args.frames):
    out, state, inter = step(mrt._params, block, constants, state)
    inter = jax.device_get(inter)

    # --- conditioning and encoder ------------------------------------------
    put("cond", np.asarray(block.values)[0, 0].astype(np.int32))           # [144]
    br = "depthformer/encoder/body/layers_1"
    # The MusicCoCa branch isn't captured on its own; the combined output is
    # (mulan + regular) / 2, so a test can recover mulan = 2*combined - regular.
    put("enc/regular", seq_values(calls(inter, br + "/branched_regular_embedder/regular_embedder")[0])[0, 0])
    put("enc/combined", seq_values(calls(inter, br, "step_with_emits")[0])[0, 0])   # before LayerNorm
    put("enc/out", seq_values(calls(inter, "depthformer/encoder/body")[0])[0, 0])  # [256]

    # --- temporal decoder ------------------------------------------------
    put("temporal/in", np.asarray(calls(inter, dec, "temporal_inputs")[0])[0, 0])   # [1024]
    # A layer's output is the output of its last sub-block, the FFN residual.
    tl = f"{dec}/temporal_body/transformer"
    layers = [seq_values(calls(inter, f"{tl}/x_layers_{i}/ffn", "step_with_emits")[0])[0, 0]
              for i in range(12)]
    put("temporal/layers", np.stack(layers))                                 # [12, 1024]
    # Layer 0 in finer steps, to pinpoint a mismatch inside one layer.
    put("temporal/l0_self", seq_values(calls(inter, f"{tl}/x_layers_0/self_attention", "step_with_emits")[0])[0, 0])
    put("temporal/l0_cross", seq_values(calls(inter, f"{tl}/x_layers_0/cross_attention", "step_with_emits")[0])[0, 0])
    put("temporal/out", np.asarray(calls(inter, dec, "temporal_outputs")[0])[0, 0])  # [1024]

    # --- depth decoder: one entry per RVQ level ---------------------------
    logits = [np.asarray(calls(inter, dec, "depth_logits")[k])[0, 0] for k in range(LEVELS)]
    put("depth/logits", np.stack([
        lg[NUM_RESERVED + k * CODEBOOK: NUM_RESERVED + (k + 1) * CODEBOOK]
        for k, lg in enumerate(logits)]))                                    # [12, 1024], before soft-cap
    put("depth/adapter", np.stack([seq_values(c)[0, 0] for c in
                                   calls(inter, f"{dec}/depth_body/depth_input_adapter")]))  # [12, 768]
    put("depth/final_ln", np.stack([seq_values(c)[0, 0] for c in
                                    calls(inter, f"{dec}/depth_body/final_ln")]))  # [12, 768]
    toks = np.array([int(np.asarray(calls(inter, dec, "depth_samples")[k]).reshape(-1)[0])
                     for k in range(LEVELS)], dtype=np.int32)
    put("tokens", toks)                                                      # [12], global ids

    # --- codec --------------------------------------------------------------
    put("codec/z", seq_values(calls(inter, f"{ss}/layers_0")[0]).reshape(-1))     # [256]
    if f < args.codec_frames:
      put("codec/input_layer", seq_values(calls(inter, f"{ss}/input_layer", "step_with_emits")[0])[0])
      put("codec/input_res", seq_values(calls(inter, f"{ss}/input_layers_residual_unit", "step_with_emits")[0])[0])
      put("codec/decoder_0", seq_values(calls(inter, f"{ss}/decoder_0", "step_with_emits")[0])[0])
      for b in range(1, 7):
        # decoder_1..6 run once per stereo group (left, right), same weights.
        per_group = [seq_values(c)[0] for c in calls(inter, f"{ss}/decoder_{b}", "step_with_emits")
                     if seq_values(c).ndim == 4]
        put(f"codec/decoder_{b}", np.stack(per_group[:2]))
      put("codec/spectrum", seq_values(calls(inter, f"{ss}/lookahead")[0])[0])     # [4, 480, 4]
    put("audio/int16", np.asarray(out.values)[0].astype(np.int32))           # [1920, 2]

    if f == 0:
      print(f"frame 0 done in {time.time() - t0:.1f}s (includes JIT compile)", flush=True)

  tensors = {}
  for name, frames in rec.items():
    tensors[name] = np.ascontiguousarray(np.stack(frames))
  meta = {
      "model": "mrt2_small", "frames": str(args.frames),
      "codec_frames": str(args.codec_frames), "temperature": "0 (greedy)",
      "compute_dtype": "float32", "style": "jazz piano trio",
      "style_tokens": ",".join(map(str, JAZZ_PIANO_TRIO)),
      "cfg": "musiccoca=3.0 notes=1.0 drums=1.0", "jax": jax.__version__,
  }
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  # One key holding sorted JSON: safetensors writes metadata keys in hash
  # order, which changes run to run; a single key makes the file
  # byte-identical across runs, so a checksum is enough to compare dumps.
  save_file(tensors, args.out, metadata={"info": json.dumps(meta, sort_keys=True)})
  print(f"{args.frames} frames in {time.time() - t0:.1f}s -> {args.out}")
  for name, t in tensors.items():
    print(f"  {str(t.dtype):8s} {str(list(t.shape)):24s} {name}")


if __name__ == "__main__":
  main()
