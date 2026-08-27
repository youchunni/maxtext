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
"""Tests for router-replay (forced routing) support in MaxTextTrainingEngine.

Lives under tests/post_training/ (not tests/unit/) because it imports
maxtext.training_engine.maxtext_engine, which pulls in tunix -- a dependency
only installed for post-training test environments. See
tests/unit/router_replay_test.py for the tunix-independent router-replay
coverage (moe.py unit tests, scan-support helpers, trainer integration
tests).
"""

import os
import sys
import unittest
from unittest import mock

import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.training_engine import maxtext_engine
from tests.utils.test_helpers import get_test_config_path


def _init_test_cfg(extra_args=(), **kwargs):
  """pyconfig.initialize with this file's common test defaults folded in."""
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


class RouterReplayEngineTest(unittest.TestCase):
  """Demonstrates that MaxTextTrainingEngine can accept forced router-replay
  expert decisions (router replay logits) on a TrainerPayload and thread them
  all the way through to the model's MoE layers via `train.loss_fn`.

  This does not touch the input pipeline: the payload is constructed
  directly by the caller (e.g. an RL rollout worker), exactly like
  `TrainerPayload` subclasses are meant to be used. `router_replay_gen_model_input_fn`
  is the "last-mile adapter" (via `with_gen_model_input_fn`) that maps it into
  keyword arguments, and `make_router_replay_loss_fn` (via `with_loss_fn`) is
  the loss function matching those kwarg names, wrapping `train.loss_fn`'s own
  `(model, config, data, ...)` calling convention.
  """

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def _build_engine(self, cfg, mesh):
    """Builds a MaxTextTrainingEngine around a real (unmocked) tiny Qwen3.5
    model, only mocking the checkpoint-loading step of from_pretrained -- the
    routing/model math itself is entirely real."""
    real_model = models.Transformer(config=cfg, mesh=mesh, quant=None, model_mode="train", rngs=nnx.Rngs(42))
    with mock.patch.object(maxtext_engine.model_creation_utils, "from_pretrained", return_value=real_model):
      engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
    return engine

  def test_engine_accepts_router_replay_payload_and_loss_is_finite(self):
    seq_len, batch_size, top_k = 16, 2, 2

    cfg = _init_test_cfg(
        extra_args=["attention=flash"],
        **_tiny_qwen35_kwargs(
            seq_len,
            batch_size,
            num_experts=4,
            top_k=top_k,
            run_name="router_replay_engine_test",
            base_num_decoder_layers=1,
            num_decoder_layers=1,
            scan_layers=False,
        ),
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    engine = self._build_engine(cfg, mesh)
    engine.with_gen_model_input_fn(maxtext_engine.router_replay_gen_model_input_fn)
    engine.with_loss_fn(maxtext_engine.make_router_replay_loss_fn(cfg))

    tokens = jnp.array([10, 20, 30, 40] * 4, dtype=jnp.int32)[:seq_len]
    token_ids = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    token_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Synthetic router-replay logits: [batch, seq_len, top_k], with the last
    # two tokens of every sequence padded (-1) to exercise the same
    # padding-mask/scatter-safety/NaN-guard code paths as
    # tests/unit/router_replay_test.py::TrainerRouterReplayTest.
    forced_routed_experts = jnp.zeros((batch_size, seq_len, top_k), dtype=jnp.int32)
    forced_routed_experts = forced_routed_experts.at[:, :, 0].set(1)
    forced_routed_experts = forced_routed_experts.at[:, :, 1].set(3)
    forced_routed_experts = forced_routed_experts.at[:, -2:, :].set(-1)

    payload = maxtext_engine.RouterReplayTrainerPayload(
        token_ids=token_ids,
        token_mask=token_mask,
        forced_routed_experts=forced_routed_experts,
    )

    # Sanity check the adapter itself: forced_routed_experts must reach the
    # kwargs router_replay_loss_fn is called with, unmodified.
    adapted_batch = maxtext_engine.router_replay_gen_model_input_fn(payload)
    self.assertIn("forced_routed_experts", adapted_batch)
    self.assertTrue((adapted_batch["forced_routed_experts"] == forced_routed_experts).all())

    engine.compile(payload)
    engine.fwd_bwd(payload)

    self.assertEqual(len(engine._cached_losses), 1)  # pylint: disable=protected-access
    loss = engine._cached_losses[-1]  # pylint: disable=protected-access
    self.assertIsNotNone(loss)
    # train.loss_fn's (loss, aux) return is converted to a WeightedMetric by
    # _fwd_bwd_kernel (aux carries xent_sum/total_weights); .compute() reduces
    # it back to a scalar, matching how the engine itself reports "loss".
    if hasattr(loss, "compute"):
      loss = loss.compute()
    self.assertFalse(jnp.isnan(loss), "Loss must not be NaN when replaying router expert decisions")
    print(f"\n[Router Replay Engine] Computed loss with forced routing via MaxTextTrainingEngine: {loss}")

  def test_gen_model_input_fn_omits_forced_routed_experts_when_absent(self):
    """When a payload doesn't set forced_routed_experts, the adapter must not
    inject a `None` value into the batch (train.loss_fn's per-key batch-size
    decimation loop indexes every value, which would crash on None)."""
    payload = maxtext_engine.RouterReplayTrainerPayload(
        token_ids=jnp.ones((1, 4), dtype=jnp.int32),
        token_mask=jnp.ones((1, 4), dtype=jnp.int32),
    )
    batch = maxtext_engine.router_replay_gen_model_input_fn(payload)
    self.assertNotIn("forced_routed_experts", batch)


if __name__ == "__main__":
  unittest.main()
