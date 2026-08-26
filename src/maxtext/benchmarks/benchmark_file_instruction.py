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

"""Benchmark script comparing Grain ArrayRecord initialization with vs without FileInstruction pre-extraction."""

import time
from absl import app
from absl import flags
import grain.python as grain
from maxtext.input_pipeline.grain_data_processing import FileInstruction, extract_file_instructions, find_data_files
from maxtext.utils import max_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("data_pattern", None, "GCS or local ArrayRecord file pattern to benchmark", required=False)
flags.DEFINE_integer("num_simulated_workers", 32, "Number of worker sidecars to simulate concurrently")


def run_benchmark(data_pattern: str, num_workers: int):
  max_logging.log(f"Starting FileInstruction benchmark on pattern: {data_pattern} with {num_workers} simulated workers.")

  # 1. Baseline: Each worker independently discovers and reads arrayrecord headers
  t0 = time.perf_counter()
  files = find_data_files(data_pattern)
  data_sources_baseline = [grain.ArrayRecordDataSource(files) for _ in range(num_workers)]
  baseline_duration = time.perf_counter() - t0
  max_logging.log(f"[Baseline] Worker independent init duration: {baseline_duration:.4f}s")

  # 2. Optimized: Coordinator extracts FileInstructions once, workers reuse them
  t0 = time.perf_counter()
  instructions = extract_file_instructions(data_pattern)
  coord_duration = time.perf_counter() - t0
  max_logging.log(f"[Optimized] Coordinator pre-extraction duration: {coord_duration:.4f}s")

  t0 = time.perf_counter()
  data_sources_optimized = [grain.ArrayRecordDataSource(instructions) for _ in range(num_workers)]
  worker_init_duration = time.perf_counter() - t0
  max_logging.log(f"[Optimized] Worker init duration from FileInstructions: {worker_init_duration:.4f}s")

  total_optimized_duration = coord_duration + worker_init_duration
  speedup = baseline_duration / max(total_optimized_duration, 1e-6)
  max_logging.log(f"[Summary] Total Optimized duration: {total_optimized_duration:.4f}s ({speedup:.2f}x speedup)")


def main(argv):
  if len(argv) > 1 and not FLAGS.data_pattern:
    FLAGS.data_pattern = argv[1]
  if FLAGS.data_pattern:
    run_benchmark(FLAGS.data_pattern, FLAGS.num_simulated_workers)
  else:
    max_logging.log("No data_pattern supplied, skipping live GCS benchmark.")


if __name__ == "__main__":
  app.run(main)
