# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards against the b/552606153 throughput regression.

ToNNX forks the caller's nnx.Rngs into a module attribute, so it lands in the model
state and its counter is incremented on device every call. MaxText bridges one of
these per quantized DenseGeneral, so with scan_layers=False the cost is paid once per
layer: llama3-8b ended up with 672 streams and 673 extra kernel launches per step,
dropping GPU utilization from 86.0% to 76.7%.

The tests use stub backends so they run on CPU without TransformerEngine. They pin
down two properties: a backend that declares it does not draw at apply time leaves no
RNG state behind, and that state does not grow with the layer count.
"""

import unittest

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import nnx

from maxtext.layers import linears
from maxtext.layers import quantizations


class _StubDotGeneral(nn.Module):
  """Minimal stand-in for a quantized dot_general Linen module.

  Declares no variables, like TE's fp8 dense path, so the only state a bridge around
  it can contribute is the forked `Rngs` itself.
  """

  @nn.compact
  def __call__(self, inputs, kernel, dims, precision=None, **kwargs):
    del kwargs
    contracting, batch = dims
    return jax.lax.dot_general(inputs, kernel, (contracting, batch), precision=precision)


class _OptOutQuant(quantizations.Quantization):
  """A backend that never draws RNGs at apply time (as TransformerEngine does not)."""

  needs_apply_rngs = False
  quant_mode = "train"  # read by quantizations.in_serve_mode()

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _StubDotGeneral


class _DefaultQuant(quantizations.Quantization):
  """A backend that leaves `needs_apply_rngs` at its safe default of True."""

  quant_mode = "train"

  def dot_general_cls(self, mesh_axes=()):
    del mesh_axes
    return _StubDotGeneral


def _rng_state_paths(module) -> list[str]:
  """Paths of every bridged-RNG leaf reachable in the module's state."""
  return [
      ".".join(str(p) for p in path)
      for path, _ in nnx.to_flat_state(nnx.state(module))
      if any("to_nnx__rngs" in str(p) for p in path)
  ]


def _make_dense(quant, seed: int = 0) -> linears.DenseGeneral:
  return linears.DenseGeneral(
      in_features_shape=8,
      out_features_shape=4,
      quant=quant,
      rngs=nnx.Rngs(params=seed, dropout=seed + 1, aqt=seed + 2),
  )


class QuantBridgeRngStateTest(unittest.TestCase):
  """The bridged quantization wrapper must not smuggle RNG state into the model."""

  def test_opt_out_backend_leaves_no_rng_state(self):
    dense = _make_dense(_OptOutQuant())
    self.assertEqual(
        _rng_state_paths(dense),
        [],
        "A backend with needs_apply_rngs=False must not retain the bridge's forked "
        "Rngs: its counters would be incremented on device every step, once per "
        "unrolled layer. See b/552606153.",
    )

  def test_default_backend_keeps_rng_state(self):
    """Checks that the opt-out is deliberate and the default stays safe."""
    paths = _rng_state_paths(_make_dense(_DefaultQuant()))
    self.assertNotEqual(paths, [], "needs_apply_rngs defaults to True, so the Rngs must be kept.")

  def test_unquantized_dense_has_no_bridge_state(self):
    self.assertEqual(_rng_state_paths(_make_dense(None)), [])

  def test_rng_state_does_not_grow_with_layer_count(self):
    """Checks the regression's signature: state scaling with the unrolled layer count.

    A scanned decoder traces one layer body, so a per-wrapper leak stays hidden. An
    unrolled one materializes every layer, so counting across a stack catches the
    regression even if the mechanism changes.
    """
    counts = []
    for num_layers in (1, 2, 8):
      layers = [_make_dense(_OptOutQuant(), seed=i) for i in range(num_layers)]
      counts.append(sum(len(_rng_state_paths(layer)) for layer in layers))

    self.assertEqual(
        counts,
        [0, 0, 0],
        f"Bridged RNG state must not scale with the number of unrolled layers, got {counts} "
        "for 1/2/8 layers. See b/552606153.",
    )

  def test_release_rngs_is_idempotent_and_keeps_the_layer_callable(self):
    """Releasing must not break apply; the wrapped module still has to run."""
    dense = _make_dense(_OptOutQuant())
    bridge = dense.quant_dot_general
    self.assertIsNotNone(bridge)
    bridge.release_rngs()  # already released during __init__; doing it again is fine

    out = dense(jnp.ones((2, 8), jnp.float32))
    self.assertEqual(out.shape, (2, 4))
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_opt_out_and_default_agree_numerically(self):
    """Dropping the Rngs is a state change, not a numerics change."""
    x = jnp.ones((2, 8), jnp.float32)
    opt_out = _make_dense(_OptOutQuant(), seed=7)(x)
    default = _make_dense(_DefaultQuant(), seed=7)(x)
    self.assertTrue(jnp.allclose(opt_out, default), "Releasing the bridge's Rngs must not change the output.")


class QuantizationFlagTest(unittest.TestCase):
  """`needs_apply_rngs` must stay safe-by-default and correct per backend."""

  def test_base_class_defaults_to_keeping_rngs(self):
    self.assertTrue(quantizations.Quantization.needs_apply_rngs)

  def test_transformer_engine_opts_out(self):
    self.assertFalse(quantizations.TransformerEngineQuantization.needs_apply_rngs)

  def test_backends_that_may_draw_at_apply_time_keep_rngs(self):
    """AQT's config enables jax.uniform RNG, so it must never be opted out silently."""
    for cls in (
        quantizations.AqtQuantization,
        quantizations.QwixQuantization,
        quantizations.Fp8Quantization,
        quantizations.NANOOFp8Quantization,
    ):
      self.assertTrue(cls.needs_apply_rngs, f"{cls.__name__} must keep its RNGs")

  def test_every_backend_declares_the_flag(self):
    """Every backend must carry the flag, so the call sites can read it directly.

    `AqtQuantization` and `QwixQuantization` sit outside the `Quantization` hierarchy
    and declare it themselves. A backend that is missing it raises AttributeError at
    the call site rather than silently taking a default.
    """
    for cls in (
        quantizations.AqtQuantization,
        quantizations.QwixQuantization,
        quantizations.Fp8Quantization,
        quantizations.NANOOFp8Quantization,
        quantizations.TransformerEngineQuantization,
    ):
      self.assertIsInstance(cls.needs_apply_rngs, bool, f"{cls.__name__} must declare needs_apply_rngs")


if __name__ == "__main__":
  unittest.main()
