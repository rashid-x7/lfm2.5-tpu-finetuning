# Fine-Tuning Liquid AI's LFM2.5-2.6B on Free Kaggle TPU v5e-8 (8,192 Context)

A reproducible recipe to fine-tune `LiquidAI/LFM2.5-2.6B` on Kaggle's free TPU v5e-8 (8 cores, 128 GB total HBM) using PyTorch/XLA and SPMD/FSDPv2.

## Key Highlights
- **Full 8,192 Context:** Solved the 105 GB / 40 GB memory ceiling using custom chunked causal attention and native Google XLA `checkpoint_module`.
- **High Throughput:** Reaches ~13,500 tokens/sec (~4.8s/step at batch size 8).
- **Time Comparison:** A full 22,000-example dataset runs in **~3.8 hours on free TPU** (vs. **46+ hours on Colab T4**).
- **Zero CPU Fallback:** Enforced with `XLA_FALLBACK_TO_CPU=0` across all hybrid convolution and GQA attention layers.

---

## Technical Traps Solved

1. **The 105 GB / 40 GB Memory Ceiling:**
   LFM2.5 is a hybrid architecture (22 gated convolution layers + 8 GQA attention layers). Without NVIDIA FlashAttention-2, standard attention decomposes into raw FP32 matrix math on TPU, generating an 8.00 GB tensor per layer (~105 GB total). We implemented a pluggable chunked causal attention kernel registered via `ALL_ATTENTION_FUNCTIONS`.

2. **The libtpu / Pallas Deadlock:**
   Pallas FlashAttention on Kaggle forces a `libtpu` upgrade that breaks the C++ PJRT runtime, causing TPU kernel deadlocks. This recipe uses upstream stable `torch==2.8.0` and `torch_xla==2.8.0` without brittle driver upgrades.

3. **Activation Checkpointing with SPMD:**
   Standard PyTorch `torch.utils.checkpoint` is ignored by the XLA compiler, schedule-locking all 30 layers in memory. We wrap each `Lfm2DecoderLayer` with Google's native `torch_xla.distributed.fsdp.checkpoint_module` prior to `FSDPv2` sharding.

4. **SafeTensors CPU Storage Pointer Bug:**
   Fixed `RuntimeError: Attempted to access the data pointer on an invalid python storage` when saving PEFT LoRA adapters on XLA by extracting the state dict into host CPU memory prior to saving.

---

## Quickstart (Kaggle)

1. Open a new Kaggle notebook with **TPU v5e-8** accelerator and **Internet ON**.
2. Upload or copy the cells from the `.ipynb` notebook in this repository.
3. In Cell 4, set `DATASET_PATH` to your dataset (or leave it blank to run in Demo Mode).
4. Run cells 1 through 5.
