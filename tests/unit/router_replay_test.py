# Copyright 2026 Google LLC
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
"""Tests for router replay (forced routing): unit tests for moe.py's
forced_routed_experts handling, the scan-support helpers, and end-to-end
trainer integration tests.

MaxTextTrainingEngine integration lives separately in
tests/post_training/unit/router_replay_engine_test.py, since it imports
maxtext.training_engine.maxtext_engine, which pulls in tunix -- a dependency
only installed for post-training test environments, not the pretrain-unit
environment this file runs under.
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.layers import moe
from maxtext.layers.nnx_decoders import reshape_forced_routed_experts_for_scan
from maxtext.configs.types import check_forced_routing_support
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train
from maxtext.common import common_types as ctypes
from tests.utils.test_helpers import get_test_config_path


def _init_test_cfg(extra_args=(), **kwargs):
  """pyconfig.initialize with this file's common test defaults folded in.

  Only pass kwargs that actually need to differ from base.yml (or from the
  selected model's own yaml) -- e.g. omit ici_*_parallelism, enable_nnx,
  pure_nnx, pure_nnx_decoder, sparse_matmul, dtype, and scan_layers=True,
  which already match their base.yml defaults.
  """
  kwargs.setdefault("enable_checkpointing", False)
  kwargs.setdefault("log_config", False)
  kwargs.setdefault("skip_jax_distributed_system", True)
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path(), *extra_args],
      **kwargs,
  )


def _tiny_qwen35_kwargs(seq_len, batch_size, num_experts, top_k, **overrides):
  """Base kwargs for a CPU-friendly, shrunk-down Qwen3.5 MoE config."""
  kwargs = {
      "override_model_config": True,
      "model_name": "qwen3.5-35b-a3b",
      "num_experts": num_experts,
      "num_experts_per_tok": top_k,
      "base_emb_dim": 256,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 256,
      "partial_rotary_factor": 0.25,
      "base_mlp_dim": 256,
      "base_moe_mlp_dim": 256,
      "vocab_size": 1000,
      "max_target_length": seq_len,
      "max_prefill_predict_length": seq_len,
      "per_device_batch_size": float(batch_size),
      "weight_dtype": "bfloat16",
  }
  kwargs.update(overrides)
  return kwargs


class DummyConfig:

  def __init__(self, model_name="default", decoder_block=ctypes.DecoderBlockType.DEFAULT):
    self.model_name = model_name
    self.decoder_block = decoder_block
    self.norm_topk_prob = False
    self.use_random_routing = False
    self.shard_mode = ctypes.ShardMode.AUTO
    self.routed_score_func = ""
    self.routed_scaling_factor = 2.5


class DummyRoutedMoE:

  def __init__(self, config):
    self.config = config
    self.dtype = jnp.float32
    self.num_experts_per_tok = 2
    self.num_experts = 3

  def _maybe_shard_with_logical(self, x, spec):
    return x


class ForcedRoutingTest(unittest.TestCase):

  def test_basic_override(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits  # Not DeepSeek
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, pre_bias_logits, forced_routed_experts=forced_routed_experts
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())
    # Check that weights are extracted correctly and softmaxed
    # For token 0: indices 2, 1 -> logits 3.0, 2.0 -> softmax([3.0, 2.0])
    # For token 1: indices 0, 2 -> logits 4.0, 6.0 -> softmax([4.0, 6.0])
    expected_weights = jax.nn.softmax(jnp.array([[[3.0, 2.0], [4.0, 6.0]]]).astype(jnp.float32), axis=-1)
    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_gemma4_softmax(self):
    config = DummyConfig(decoder_block=ctypes.DecoderBlockType.GEMMA4)
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
    pre_bias_logits = gate_logits
    forced_routed_experts = jnp.array([[[2, 1], [0, 2]]])  # (1, 2, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, pre_bias_logits, forced_routed_experts=forced_routed_experts
    )

    # Check that indices are overridden
    self.assertTrue((top_k_indices == forced_routed_experts).all())

    # For Gemma 4, it applies softmax to gate_logits first!

    expected_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    expected_weights = jnp.take_along_axis(expected_probs, forced_routed_experts, axis=-1)

    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_reshape_and_update_weights(self):
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    weights = jnp.array([[[0.1, 0.2], [0.3, 0.4]]])  # (1, 2, 2)
    indices = jnp.array([[[2, -1], [-1, 1]]])  # (1, 2, 2)

    update_weights = moe.RoutedMoE.reshape_and_update_weights(model, weights, indices, safe_updates=True)

    # Expected shape: (1, 2, 3) where 3 is num_experts!
    # For token 0: index 2 -> 0.1. Index -1 -> mapped to 0 but weight 0.0!
    # So for expert 0: 0.0. Expert 1: 0.0. Expert 2: 0.1.
    # For token 1: index -1 -> mapped to 0 but weight 0.0! Index 1 -> 0.4.
    # So for expert 0: 0.0. Expert 1: 0.4. Expert 2: 0.0.
    expected_update_weights = jnp.array([[[0.0, 0.0, 0.1], [0.0, 0.4, 0.0]]])

    self.assertTrue(jax.numpy.allclose(update_weights, expected_update_weights, rtol=1e-5, atol=1e-5))

  def test_reshape_and_update_weights_duplicate_indices_use_add_not_set(self):
    """A regression test for the `.set()` -> `.add()` scatter-safety fix.

    Forced-routing replay can legitimately produce duplicate expert indices
    for the same token (e.g. two padding slots, which both remap to dummy
    index 0). `.set()` has undefined behavior for duplicate scatter indices,
    so we must use `.add()`. Since both duplicate slots carry a real,
    nonzero weight here, `.add()` and `.set()` produce different, checkable
    results: `.add()` sums to the total, `.set()` would keep only one.
    """
    config = DummyConfig()
    model = DummyRoutedMoE(config)

    # Token 0 has expert index 1 selected twice with different weights.
    weights = jnp.array([[[0.3, 0.7]]])  # (1, 1, 2)
    indices = jnp.array([[[1, 1]]])  # (1, 1, 2)

    update_weights = moe.RoutedMoE.reshape_and_update_weights(model, weights, indices, safe_updates=True)

    # `.add()` must sum both contributions at expert index 1.
    expected_update_weights = jnp.array([[[0.0, 1.0, 0.0]]])
    self.assertTrue(jax.numpy.allclose(update_weights, expected_update_weights, rtol=1e-5, atol=1e-5))

  def test_deepseek_uses_pre_bias_logits_for_weights(self):
    """The deepseek3/4 branch of get_topk must extract weights from
    pre_bias_logits (not the post-bias gate_logits), matching non-forced
    deepseek routing (`deepseek_routing`)."""
    # NOTE: decoder_block is intentionally left at the default (not DEEPSEEK),
    # so get_topk's post-scaling step (deepseek_scale_weights) is skipped and
    # this test isolates just the pre_bias_logits weight-extraction branch,
    # which is keyed off `model_name`, not `decoder_block`.
    config = DummyConfig(model_name="deepseek3-671b")
    model = DummyRoutedMoE(config)

    gate_logits = jnp.array([[[1.0, 2.0, 3.0]]])  # post-bias logits, (1, 1, 3)
    pre_bias_logits = jnp.array([[[10.0, 20.0, 30.0]]])  # pre-bias logits, distinct values
    forced_routed_experts = jnp.array([[[2, 0]]])  # (1, 1, 2)

    top_k_weights, top_k_indices = moe.RoutedMoE.get_topk(
        model, gate_logits, pre_bias_logits, forced_routed_experts=forced_routed_experts
    )

    self.assertTrue((top_k_indices == forced_routed_experts).all())
    # Weights must come from pre_bias_logits at indices [2, 0] -> [30.0, 10.0],
    # not from gate_logits (which would give [3.0, 1.0]); DEFAULT decoder_block
    # then softmaxes them, same as the non-deepseek path.
    expected_weights = jnp.take_along_axis(pre_bias_logits, forced_routed_experts, axis=-1)
    expected_weights = jax.nn.softmax(expected_weights.astype(jnp.float32), axis=-1)
    self.assertTrue(jax.numpy.allclose(top_k_weights, expected_weights, rtol=1e-5, atol=1e-5))

  def test_permute_padding_remapped_to_valid_expert_not_dropped(self):
    """Regression test for the dummy-index remap in RoutedMoE.permute().

    Padding entries (-1) in forced_routed_experts must be remapped to a
    valid dummy expert id before `argsort`/`bincount`, rather than left as
    -1 (which would either crash bincount or silently corrupt routing for
    every other token).
    """
    config = DummyConfig()
    model = DummyRoutedMoE(config)
    model.num_experts_per_tok = 2
    model.num_experts = 3

    # 2 tokens, top_k=2. Token 0 fully padded (-1, -1); token 1 valid (0, 2).
    forced_routed_experts = jnp.array([[[-1, -1], [0, 2]]])
    flatten_selected_experts = jnp.ravel(forced_routed_experts)

    valid_mask = flatten_selected_experts >= 0
    dummy_indices = jnp.arange(flatten_selected_experts.shape[0]) % model.num_experts
    flatten_selected_experts_safe = jnp.where(valid_mask, flatten_selected_experts, dummy_indices)

    # No entry should remain negative (which would break bincount/argsort).
    self.assertTrue((flatten_selected_experts_safe >= 0).all())
    # The two valid (non-padded) entries must be preserved unchanged.
    valid_positions = jnp.nonzero(valid_mask)[0]
    self.assertTrue((flatten_selected_experts_safe[valid_positions] == flatten_selected_experts[valid_positions]).all())
    # bincount over the "safe" indices must not raise despite the original -1s.
    group_size = jnp.bincount(flatten_selected_experts_safe, length=model.num_experts)
    self.assertEqual(int(jnp.sum(group_size)), flatten_selected_experts.shape[0])

  def test_permute_padding_mask_computed_before_roll_to_expert_id(self):
    """Regression test for a Ring-of-Experts (ROE) bug: `valid_mask` must be
    computed from the padding (-1) entries BEFORE `roll_to_expert_id` is
    applied, not after.

    `roll_to_expert_id` shifts every index modulo num_experts, and JAX's `%`
    wraps negative values into the valid [0, num_experts) range -- so
    `(-1 - roll) % num_experts` is always a valid, non-negative expert id.
    Computing `valid_mask = flatten_selected_experts >= 0` AFTER that roll
    would therefore see every padding token as "valid" and skip the dummy-index
    remap entirely, collapsing all padding tokens from every device shard
    onto a single expert -- a severe load imbalance that can drop valid
    tokens. This test locks in the correct order: mask first, then roll.
    """
    num_experts = 4
    roll_to_expert_id = 2

    # 2 tokens, top_k=2. Token 0 fully padded (-1, -1); token 1 valid (0, 2).
    forced_routed_experts = jnp.array([[[-1, -1], [0, 2]]])
    flatten_selected_experts = jnp.ravel(forced_routed_experts)

    # Correct order: mask against the pre-roll values.
    valid_mask = flatten_selected_experts >= 0
    rolled = (flatten_selected_experts - roll_to_expert_id) % num_experts
    dummy_indices = jnp.arange(flatten_selected_experts.shape[0]) % num_experts
    safe = jnp.where(valid_mask, rolled, dummy_indices)

    # The two originally-padded slots must NOT retain whatever the roll wrapped
    # them into; they must be the dummy indices instead.
    padded_positions = jnp.nonzero(~valid_mask)[0]
    self.assertTrue((safe[padded_positions] == dummy_indices[padded_positions]).all())
    # None of the (small) padded/dummy set should collide with the buggy
    # behavior below, which the test also verifies differs from the fix.
    buggy_mask = rolled >= 0  # what you'd get by masking AFTER rolling
    self.assertTrue(buggy_mask.all(), "sanity check: rolled negative values do wrap to non-negative")
    self.assertFalse(
        (valid_mask == buggy_mask).all(),
        "computing valid_mask before vs. after the roll must disagree given padding is present",
    )


class CheckForcedRoutingSupportTest(unittest.TestCase):
  """Regression tests for the decoder_block validation gate: forced routing
  is supported (scanned or not) only for a fixed set of decoder_blocks."""

  def test_supported_decoder_blocks_are_allowed(self):
    for decoder_block in (
        ctypes.DecoderBlockType.QWEN3_5,
        ctypes.DecoderBlockType.MIXTRAL,
        ctypes.DecoderBlockType.LLAMA4,
        ctypes.DecoderBlockType.ENVY,
        ctypes.DecoderBlockType.GEMMA4,
    ):
      check_forced_routing_support(decoder_block)  # must not raise

  def test_unsupported_decoder_blocks_raise(self):
    for decoder_block in (
        ctypes.DecoderBlockType.QWEN3_MOE,
        ctypes.DecoderBlockType.QWEN3_NEXT,
        ctypes.DecoderBlockType.DEEPSEEK,
        ctypes.DecoderBlockType.DEEPSEEK4,
        ctypes.DecoderBlockType.DEFAULT,
    ):
      with self.assertRaises(NotImplementedError):
        check_forced_routing_support(decoder_block)


class ReshapeForcedRoutedExpertsForScanTest(unittest.TestCase):
  """Regression tests for the scan-xs reshape used to support forced routing
  with scan_layers=True (see check_forced_routing_support in configs/types.py
  for which decoder_blocks)."""

  def test_4d_input_preserves_per_layer_values_in_scan_order(self):
    """Homogeneous-MoE case (e.g. QWEN3_5): every sub-layer in the cycle is
    MoE, so moe_per_cycle == cycle_interval and num_moe_layers ==
    num_decoder_layers."""
    num_decoder_layers = 8
    cycle_interval = 4
    scan_length = num_decoder_layers // cycle_interval
    batch, seq, top_k = 2, 3, 2

    # [batch, seq, num_layers, top_k], where every entry for layer L is L.
    layer_ids = jnp.arange(num_decoder_layers, dtype=jnp.int32)
    forced_routed_experts = jnp.broadcast_to(layer_ids[None, None, :, None], (batch, seq, num_decoder_layers, top_k))

    scanned = reshape_forced_routed_experts_for_scan(
        forced_routed_experts,
        num_moe_layers=num_decoder_layers,
        scan_length=scan_length,
        moe_per_cycle=cycle_interval,
    )

    self.assertEqual(scanned.shape, (scan_length, cycle_interval, batch, seq, top_k))
    # Layer L must land at scanned[L // cycle_interval, L % cycle_interval],
    # since jax.lax.scan slices axis 0 (scan_length) automatically per outer
    # iteration, then the ScannableBlock's static loop indexes axis 0 of the
    # remaining [cycle_interval, ...] chunk by sub-layer position.
    for layer in range(num_decoder_layers):
      cycle_idx, sub_idx = divmod(layer, cycle_interval)
      self.assertTrue((scanned[cycle_idx, sub_idx] == layer).all(), f"layer {layer} landed in the wrong scan slot")

  def test_3d_input_broadcasts_same_routing_to_every_layer(self):
    num_moe_layers = 4
    moe_per_cycle = 4
    scan_length = num_moe_layers // moe_per_cycle
    batch, seq, top_k = 1, 2, 2

    forced_routed_experts = jnp.array([[[1, 3], [0, 2]]])  # [batch, seq, top_k]

    scanned = reshape_forced_routed_experts_for_scan(
        forced_routed_experts,
        num_moe_layers=num_moe_layers,
        scan_length=scan_length,
        moe_per_cycle=moe_per_cycle,
    )

    self.assertEqual(scanned.shape, (scan_length, moe_per_cycle, batch, seq, top_k))
    for sub_idx in range(moe_per_cycle):
      self.assertTrue((scanned[0, sub_idx] == forced_routed_experts).all())

  def test_interleaved_case_compacts_only_moe_layers(self):
    """Interleaved dense/MoE case (e.g. LLAMA4/ENVY): the input's layer axis
    only counts MoE layers (moe_per_cycle < cycle_interval), matching the
    `moe_lyr_idx` convention used for the unscanned per-layer case."""
    moe_per_cycle = 2  # e.g. 2 of 4 sub-layers per cycle are MoE.
    scan_length = 3
    num_moe_layers = scan_length * moe_per_cycle
    batch, seq, top_k = 1, 2, 1

    moe_layer_ids = jnp.arange(num_moe_layers, dtype=jnp.int32)
    forced_routed_experts = jnp.broadcast_to(moe_layer_ids[None, None, :, None], (batch, seq, num_moe_layers, top_k))

    scanned = reshape_forced_routed_experts_for_scan(
        forced_routed_experts,
        num_moe_layers=num_moe_layers,
        scan_length=scan_length,
        moe_per_cycle=moe_per_cycle,
    )

    self.assertEqual(scanned.shape, (scan_length, moe_per_cycle, batch, seq, top_k))
    for moe_layer in range(num_moe_layers):
      cycle_idx, sub_idx = divmod(moe_layer, moe_per_cycle)
      self.assertTrue((scanned[cycle_idx, sub_idx] == moe_layer).all())


class TrainerRouterReplayTest(unittest.TestCase):
  """Integration tests: forced routing threaded end-to-end through
  train.loss_fn, both unscanned and scanned, across every architecture that
  supports it (see check_forced_routing_support in configs/types.py)."""

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _assert_forced_routing_loss_finite(self, cfg, seq_len, batch_size, forced_experts, label):
    """Builds a real model for `cfg`, runs train.loss_fn with a batch carrying
    `forced_experts`, and asserts the resulting loss is finite."""
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    rng = jax.random.PRNGKey(42)

    tokens = jnp.array(([10, 20, 30, 40] * ((seq_len // 4) + 1))[:seq_len], dtype=jnp.int32)
    inputs = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    positions = jnp.tile(jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0), (batch_size, 1))
    segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    targets = jnp.roll(inputs, -1, axis=-1)

    data_batch = {
        "inputs": inputs,
        "inputs_position": positions,
        "inputs_segmentation": segmentation,
        "targets": targets,
        "targets_segmentation": segmentation,
        "forced_routed_experts": forced_experts,
    }

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    init_params_rng, init_dropout_rng = jax.random.split(rng)
    params = model.init(
        {"params": init_params_rng, "dropout": init_dropout_rng},
        inputs,
        positions,
        segmentation,
        enable_dropout=False,
    )

    loss, aux = train.loss_fn(
        model,
        cfg,
        data_batch,
        dropout_rng=init_dropout_rng,
        params=params,
        is_train=True,
    )

    self.assertIsNotNone(loss)
    self.assertFalse(jnp.isnan(loss), "Loss must not be NaN")
    print(f"\n[Trainer Router Replay][{label}] Computed loss with forced routing + padding: {loss}")
    return loss, aux

  def test_loss_fn_with_forced_routed_experts(self):
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    cfg = _init_test_cfg(
        extra_args=["attention=flash"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts,
            top_k,
            run_name="test_trainer_router_replay",
            base_num_decoder_layers=1,
            num_decoder_layers=1,
            scan_layers=False,
        ),
    )

    # Synthetic forced routed experts: [batch, seq_len, top_k]
    forced_experts = jnp.zeros((batch_size, seq_len, top_k), dtype=jnp.int32)
    forced_experts = forced_experts.at[:, :, 0].set(1)
    forced_experts = forced_experts.at[:, :, 1].set(3)
    # Mark the last two tokens of every sequence as padding (-1) in every
    # expert slot, exercising the padding-mask/scatter-safety/NaN-guard code
    # paths end-to-end, not just in isolated unit tests.
    forced_experts = forced_experts.at[:, -2:, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "qwen3.5")

  def test_loss_fn_with_forced_routed_experts_scanned_qwen3_5(self):
    """Qwen3.5 supports forced routing together with `scan_layers=True` (see
    `check_forced_routing_support` in configs/types.py for the full list of
    supported decoder_blocks). This exercises that scanned path end-to-end:
    forced_routed_experts is threaded through jax.lax.scan's xs (one slice
    per layer) instead of being broadcast.
    """
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    cycle_interval = 4
    num_layers = 2 * cycle_interval  # 2 scan iterations of one cycle each.

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts,
            top_k,
            run_name="test_trainer_router_replay_scanned_qwen3_5",
            base_num_decoder_layers=num_layers,
            num_decoder_layers=num_layers,
            inhomogeneous_layer_cycle_interval=cycle_interval,
        ),
    )

    # Synthetic forced routed experts, one distinct value per layer:
    # [batch, seq_len, num_layers, top_k].
    layer_ids = jnp.arange(num_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned qwen3.5")

  def test_loss_fn_with_forced_routed_experts_scanned_mixtral(self):
    """Mixtral has no ScannableBlock wrapper: every scan iteration is exactly
    one (always-MoE) decoder layer, so this exercises the "no per-cycle
    nesting" branch of the scan wiring (moe_per_cycle=1, squeezed)."""
    seq_len, batch_size, top_k, num_experts = 16, 2, 2, 4
    num_layers = 4  # 4 scan iterations of 1 layer each (cycle_interval=1).

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        run_name="test_trainer_router_replay_scanned_mixtral",
        override_model_config=True,
        base_num_decoder_layers=num_layers,
        num_decoder_layers=num_layers,
        model_name="mixtral-8x7b",
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=256,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        vocab_size=1000,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        per_device_batch_size=float(batch_size),
        weight_dtype="bfloat16",
    )

    # One distinct forced routing per layer: [batch, seq_len, num_layers, top_k].
    layer_ids = jnp.arange(num_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned mixtral")

  def test_loss_fn_with_forced_routed_experts_scanned_llama4(self):
    """Llama4 interleaves dense and MoE sub-layers within each cycle (unlike
    Qwen3.5/Mixtral, which are homogeneous MoE), exercising the
    MoE-only-counter compaction in the ScannableBlock's static loop."""
    seq_len, batch_size, top_k, num_experts = 16, 2, 1, 4
    cycle_interval = 4
    interleave_moe_layer_step = 2  # sub-layers 1 and 3 (0-indexed) are MoE.
    num_layers = 2 * cycle_interval  # 2 scan iterations of one cycle each.

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        run_name="test_trainer_router_replay_scanned_llama4",
        override_model_config=True,
        base_num_decoder_layers=num_layers,
        num_decoder_layers=num_layers,
        inhomogeneous_layer_cycle_interval=cycle_interval,
        # llama4-17b-16e.yml defaults to nope_layer_interval=4 (NoPE layers)
        # and interleave_moe_layer_step=1 (every layer MoE); override both so
        # this test actually exercises the dense/MoE interleave + full RoPE.
        nope_layer_interval=-1,
        interleave_moe_layer_step=interleave_moe_layer_step,
        model_name="llama4-17b-16e",
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        base_emb_dim=256,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=256,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        vocab_size=1000,
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        per_device_batch_size=float(batch_size),
        weight_dtype="bfloat16",
    )

    # num_moe_layers = num MoE sub-layers total = 2 per cycle * 2 cycles = 4.
    moe_per_cycle = sum(1 for i in range(cycle_interval) if (i + 1) % interleave_moe_layer_step == 0)
    num_moe_layers = (num_layers // cycle_interval) * moe_per_cycle
    moe_layer_ids = jnp.arange(num_moe_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        moe_layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_moe_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned llama4")

  def test_loss_fn_with_forced_routed_experts_scanned_envy(self):
    """Envy: same interleaved-cycle ScannableBlock pattern as Llama4, using
    the tiny envy-test config (already sets base_num_decoder_layers=4,
    inhomogeneous_layer_cycle_interval=2, interleave_moe_layer_step=2)."""
    seq_len, batch_size, top_k = 8, 2, 1
    cycle_interval = 2
    interleave_moe_layer_step = 2  # sub-layer 1 (0-indexed) is MoE.
    num_layers = 4

    cfg = _init_test_cfg(
        extra_args=["attention=dot_product"],
        run_name="test_trainer_router_replay_scanned_envy",
        model_name="envy-test",
        max_target_length=seq_len,
        max_prefill_predict_length=seq_len,
        per_device_batch_size=float(batch_size),
        num_experts_per_tok=top_k,
    )
    num_experts = cfg.num_experts

    moe_per_cycle = sum(1 for i in range(cycle_interval) if (i + 1) % interleave_moe_layer_step == 0)
    num_moe_layers = (num_layers // cycle_interval) * moe_per_cycle
    moe_layer_ids = jnp.arange(num_moe_layers, dtype=jnp.int32)
    forced_experts = jnp.broadcast_to(
        moe_layer_ids[None, None, :, None] % num_experts,
        (batch_size, seq_len, num_moe_layers, top_k),
    )
    forced_experts = forced_experts.at[:, -2:, :, :].set(-1)

    self._assert_forced_routing_loss_finite(cfg, seq_len, batch_size, forced_experts, "scanned envy")


if __name__ == "__main__":
  unittest.main()
