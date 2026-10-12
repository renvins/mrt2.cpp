"""Check the model spec (plan section 2) against a reference dump.

Every fact the C++ relies on is recomputed here in numpy, from the raw
weights, and compared with what Google's model actually produced. If one
fails, the plan's spec is wrong, and C++ written from it would be too.

Usage (from the mrt2 folder):
  .venv/bin/python tools/check_spec.py REF.safetensors MODEL.safetensors
"""
import sys

import numpy as np
from safetensors.numpy import load_file
R = load_file(sys.argv[1]); W = load_file(sys.argv[2])
P = "params/depthformer/"
FAILED = []
def ok(c, msg):
    print(("ok   " if c else "FAIL ") + msg)
    if not c: FAILED.append(msg)
T = R["tokens"]; F = T.shape[0]
# 1. token ids: level k is in [6+1024k, 6+1024(k+1)), and greedy = argmax of the logit slice
ok(all(6+1024*k <= T[f,k] < 6+1024*(k+1) for f in range(F) for k in range(12)), "token ids: 6 + level*1024 + code")
ok(np.array_equal(T - 6 - 1024*np.arange(12), R["depth/logits"].argmax(-1)), "greedy token = argmax of logit slice (soft-cap doesn't change argmax)")
# 2. conditioning layout
c = R["cond"][0]
ok(list(c[:12]) == [x+7 for x in [555,402,72,424,482,932,954,153,33,374,519,779]], "cond[0:12] = style tokens + 7")
ok((c[12:141] == 6).all(), "piano roll + drum unset = -1 + 7 = 6")
ok(list(c[141:144]) == [27, 17, 9], "CFG tokens = 20+7, 10+7, 2+7")
# 3. temporal input = mean(32 * E[previous tokens]), frame 0 uses SOS = 0
E = W[P+"decoder/decoder_embedding/embedding/embedding"]
prev = [np.zeros(12, int)] + [T[f-1] for f in range(1, F)]
ti = np.stack([(32*E[p]).mean(0) for p in prev])
ok(np.abs(ti - R["temporal/in"]).max() < 1e-5, f"temporal input = mean(32*E[prev tokens]), SOS=0  (max err {np.abs(ti - R['temporal/in']).max():.2e})")
# 4. no final norm: temporal output == last layer's output
ok(np.array_equal(R["temporal/out"], R["temporal/layers"][:, 11]), "temporal output = layer 11 output exactly (no final norm)")
# 5. encoder: LayerNorm(combined) with eps 1e-6
x = R["enc/combined"]; mu = x.mean(-1, keepdims=True); var = ((x-mu)**2).mean(-1, keepdims=True)
ln = (x-mu)/np.sqrt(var+1e-6)*W[P+"encoder/body/encoder_ln/scale"] + W[P+"encoder/body/encoder_ln/bias"]
ok(np.abs(ln - R["enc/out"]).max() < 1e-5, f"enc/out = LayerNorm(combined), eps 1e-6  (max err {np.abs(ln-R['enc/out']).max():.2e})")
# 6. encoder regular branch = mean over 132 channel rows, offsets = cumulative vocab sizes
sizes = [11]*128 + [9] + [47]*2 + [15]; off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
RE = W[P+"encoder/body/layers_1/branched_regular_embedder/regular_embedder/embedding"]
reg = RE[off + c[12:144]].mean(0)
ok(np.abs(reg - R["enc/regular"][0]).max() < 1e-5, f"regular branch = mean of 132 rows at cumulative offsets  (max err {np.abs(reg-R['enc/regular'][0]).max():.2e})")
# 7. MusicCoCa branch = Dense(sum of 12 rows at level*1031 + tok); combined = (mulan + regular) / 2
ME = W[P+"encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding"]
A = W[P+"encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel"]
mul = ME[np.arange(12)*1031 + c[:12]].sum(0) @ A
ok(np.abs((mul + reg)/2 - R["enc/combined"][0]).max() < 1e-5, f"combined = (mulan + regular)/2, mulan = sum(12 rows) @ adapter  (max err {np.abs((mul+reg)/2-R['enc/combined'][0]).max():.2e})")
# 8. codec input z = sum over 12 levels of Q[k][code]; which frame's tokens?
Q = W["params/soundstream/quantizer/embedding"]
z_same = np.stack([sum(Q[k][(T[f,k]-6) % 1024] for k in range(12)) for f in range(F)])
ok(np.abs(z_same - R["codec/z"]).max() < 1e-5, f"codec z[t] = sum_k Q[k][code(t,k)] (same frame)  (max err {np.abs(z_same-R['codec/z']).max():.2e})")
# 9. codec lookahead: the first frame's audio is silent, the next ones are not
a = R["audio/int16"]
ok(np.abs(a[0]).max() == 0 and np.abs(a[1]).max() > 0, "audio frame 0 is silent (1-frame codec lookahead)")
sys.exit(1 if FAILED else 0)
