# Training: the printer stories, the TONER-1 fine-tune and TONER-1 Pro

This folder is the pipeline behind the shipped weights:

1. **Printer stories.** 9,218 short children's stories about printers, paper, ink and toner, in the style of
   TinyStories, written by a free model on OpenRouter and filtered (`data/`).
2. **TONER-1.** Karpathy's `stories260K` fine-tuned on those stories, with TinyStories replay
   (`weights/TONER-1/` at the repo root).
3. **TONER-1 Pro.** Our own 4,096-token tokenizer (`tokenizer/`) and a 12,194,688-parameter model trained
   from scratch on TinyStories, then fine-tuned on the same printer stories (`weights/TONER-1-Pro/`).

You do not need any of this to rebuild the print jobs: `sh tools/rebuild.sh` at the repo root does that
from the committed weights. Training on Apple's GPU (MPS) is not bit-reproducible, and the story generator
is a sampled LLM, so a rerun gives similar models, not the same bytes. That is why the outputs are
committed. The deterministic steps (dataset tokenization, the tokenizer, the exports) do reproduce byte
for byte; the MD5s are below.

## What is here

| Path | What it is |
|---|---|
| `scripts/paths.py` | Where everything is read and written. The work directory is `$TONER_TRAIN`, default `training/work/` (git ignores it) |
| `scripts/gen_stories.py`, `scripts/or_free.py` | The story generator and its OpenRouter client (free models only, with $0 guards). No key is in this repo |
| `scripts/build_vocab.py`, `scripts/build_datasets.py`, `scripts/textnorm.py` | TinyStories word counts for the simple-word filter; the final filters and the 95/5 split; ASCII normalization |
| `scripts/cost_check.py` | Looks up every logged generation id and sums its cost (it was $0) |
| `scripts/dump_prompt_lists.py` | Writes `data/prompt_lists.json` from `gen_stories.py` |
| `scripts/ft260_*.py`, `scripts/check_260k.py` | The TONER-1 fine-tune: tokenization, training, export to llama2.c's `.bin`, evaluation |
| `scripts/pro_tokenizer.py` | Trains the Pro tokenizer and writes `tok4096.model` and `tok4096.bin` |
| `scripts/pro_pretokenize.py`, `scripts/pro_train.py`, `scripts/pro_finalize.py` | Pro pretraining with llama2.c's `train.py` |
| `scripts/pro_ft_*.py` | The Pro printer fine-tune, its export and its evaluation |
| `configs/` | `pro_B.py` (the llama2.c config file for pretraining) and the recorded settings of the three shipped runs |
| `llama2c/` | llama2.c's `model.py`, `export.py`, `configurator.py`, and `train.py` and `tinystories.py` with our patches (`PATCH_*.diff`). MIT. Its `tokenizer.py` is left out (see `COMMIT.txt`) |
| `tokenizer/` | The Pro tokenizer: `tok4096.model` (SentencePiece), `tok4096.bin` (llama2.c format, what the print job embeds), `tok4096.vocab`, `tok4096.json` |
| `data/printer_train.txt`, `data/printer_val.txt` | The printer stories, 8,757 train and 461 validation, in TinyStoriesV2's layout (each story, then a line `<\|endoftext\|>`) |
| `data/pilot_printer_stories_601.jsonl` | The first 601 generated stories (id and text), used only to train the tokenizer |
| `data/prompt_lists.json` | The generator's prompt ingredients: system prompt, 68 themes, word lists, features, filters, one example prompt |

## Setup

Python 3.9 or later with `torch` (2.8 was used, on an Apple M4 with MPS), `numpy` and `sentencepiece`.
Run everything from the repo root. Everything downloaded or produced goes to `$TONER_TRAIN`.

```sh
export TONER_TRAIN=$PWD/training/work            # the default; any folder works
sh tools/fetch_checkpoints.sh --train             # stories260K.pt, stories260K.bin, tok512.model, tok512.bin -> checkpoints/
mkdir -p $TONER_TRAIN/data/tinystories && cd $TONER_TRAIN/data/tinystories
curl -LO https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt   # 2.2 GB
curl -LO https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
shasum -a 256 TinyStoriesV2-GPT4-*.txt
#   6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443  TinyStoriesV2-GPT4-train.txt
#   6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d  TinyStoriesV2-GPT4-valid.txt
cd - && python -I training/scripts/build_vocab.py   # word counts for the simple-word filter and the metrics
```

## 1. The printer stories (optional: the result is committed)

The generator was **Xing4.0-29B-A4B** by China Telecom's TeleAI (Apache-2.0,
<https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B>), through OpenRouter's free tier as
`xingchen-agi/xing4.0-29b-a4b:free`, provider Phala, at temperature 0.8, top_p 0.95, reasoning off.

Each request asks for 5 to 12 stories. Each story gets a printer theme (68 themes, kept even), a required
noun, verb and adjective from simple-word lists, one or two story features, sometimes a character name,
and in 70% of stories one of TONER-1-V's 18 drawable things. That is the TinyStories recipe (Eldan and Li
2023) with printer themes. No TinyStories text is in the prompts. Replies are parsed and filtered: ASCII
only, no markup, 80 to 220 words, at least 3 printer, paper, ink or toner words, at most 2 words outside
TinyStories' frequent words plus a printer whitelist, and no near-duplicates.

The key is read inside the process from `OPENROUTER_API_KEY`, or else from a key file
(`$OPENROUTER_KEY_FILE`, default `training/.env`, ignored by git). It is never printed or logged. The
client refuses any model whose id does not end in `:free` or whose listed price is not zero, and stops if a
response reports a cost.

```sh
python -I training/scripts/gen_stories.py --dry-run                        # print one prompt; no network
python -I training/scripts/gen_stories.py --target 12000 --per-request 12 --workers 6 --max-seconds 410 \
    --retry-sleep 5 --max-attempts 40
python -I training/scripts/build_datasets.py                               # -> $TONER_TRAIN/data/printer_{train,val}.{txt,json}
python -I training/scripts/cost_check.py                                    # sums the cost of every generation id
```

`gen_stories.py` is resumable, and the bulk run was about 30 chunks like the second command (most with these
arguments). The prompt
changed during the run (versions v2 to v8, mostly about length; see `PROMPT_VERSION` in the script);
`build_datasets.py` re-applies the final filters to everything. The result: 1,083 requests, 983 answered,
11,003 stories asked for, 9,218 kept (8,757 train, 461 validation, seeded split), 118 words on average,
$0.00 in total. To train on a new set instead of the committed one, set `PRINTER_DATA=$TONER_TRAIN/data`.

## 2. TONER-1: fine-tune stories260K

```sh
python -I training/scripts/ft260_prep.py
python -I training/scripts/ft260_train.py --name D_lr3e-4_mix70 --lr 3e-4 --iters 3000 --printer_frac 0.7
python -I training/scripts/ft260_train.py --name D_lr3e-4_mix70 --lr 3e-4 --iters 3000 --printer_frac 0.7 --resume
python -I training/scripts/ft260_export.py $TONER_TRAIN/ckpt/ft260/D_lr3e-4_mix70/iter_03000.pt build/toner1.bin
python3 tools/l2c_safetensors.py to-safetensors build/toner1.bin weights/TONER-1 "TONER-1" "tok512: ..."
```

`ft260_prep.py` tokenizes with Karpathy's `tok512`, unchanged: the printer stories, every 18th TinyStoriesV2
train story as replay (150,957 stories, 59.1M tokens), and 2,500 held-out TinyStoriesV2 validation stories.
Its outputs reproduce exactly: `printer_train.bin` 273b6e1a56b1570628a557c0055c58a4, `printer_val.bin`
1eba6caeb43020324bd85e039a0145a8, `ts_replay.bin` dfc9a982f6710f33d66089805a3e6146, `ts_val.bin`
4997d961081965b194042e2271498149.

Run D was the one we shipped, out of six (A to F: learning rate 1e-4 or 3e-4, printer share 50% to 80%,
3,000 or 6,000 steps). Each step is 16 rows of 512 tokens, 11 printer and 5 replay; 3,000 steps at lr 3e-4
with 100 warmup steps and cosine decay to 3e-5; AdamW (0.9, 0.99), weight decay 0.01, gradient clip 1.0,
dropout 0, seed 1337. It took about 11 minutes on the M4, in two chunks (`--max_minutes 8`, then
`--resume`). The settings are in `configs/ft260_run_D.json`. `ft260_export.py` writes llama2.c's legacy
`.bin` (MD5 7c98049fc9e4c05e2bbc7281af73364a) and checks that it gives exactly the checkpoint's logits.

| int8, as the print job runs it | stories260K | TONER-1 |
|---|---|---|
| TinyStories validation loss | 1.3411 | 1.3934 |
| Printer validation loss (461 stories) | 1.8286 | 1.2270 |
| Samples with any printer word (n = 300, T 0.8, 200 tokens) | 3.7% | 87.3% |
| Samples that say "printer" | 0% | 71.7% |

Evaluate with `python -I training/scripts/ft260_eval.py --label D CKPT.pt --int8 --n 300`.

## 3. TONER-1 Pro: the tokenizer

```sh
python -I training/scripts/pro_tokenizer.py --vocab_size=4096 --sample=300000 --printer_repeat=60 \
    --user_symbols=▁toner,▁Toner
cmp $TONER_TRAIN/tokenizer/tok4096.model training/tokenizer/tok4096.model
cmp $TONER_TRAIN/tokenizer/tok4096.bin training/tokenizer/tok4096.bin
```

SentencePiece BPE with llama2.c's `train_vocab` settings, trained on a seeded sample of 299,945
TinyStoriesV2 stories plus the 601 pilot printer stories repeated 60 times (8.5% of the words), so that
printer words get their own tokens. `▁toner` and `▁Toner` are forced in as user-defined pieces. The
`.bin` export is written by our own code, not llama2.c's `tokenizer.py`. A rerun on the M4 is byte-identical
to the committed files. SentencePiece records the machine's thread count in the `.model`, so on a machine
with a different core count that one field differs; the pieces, the scores and the `.bin` do not.
`tok4096.bin` MD5 7701c417195143790f7f5241bf6a1050; `tok4096.model` MD5 7edb2c3a80f746e0aabe5692733a11bc.

## 4. TONER-1 Pro: pretraining

```sh
python -I training/scripts/pro_pretokenize.py           # TinyStoriesV2 -> $TONER_TRAIN/data/tok4096/data00..28.bin
python -I -u training/scripts/pro_train.py training/configs/pro_B.py                      # first chunk
python -I -u training/scripts/pro_train.py training/configs/pro_B.py --init_from=resume   # repeat to max_iters
python -I training/scripts/pro_finalize.py              # -> $TONER_TRAIN/ckpt/pro/pretrain.pt, pretrain_v0.bin
```

dim 384, 6 layers, 6 heads (6 KV heads), feed-forward 1024, vocabulary 4,096, context 256: 12,194,688
parameters. 3,800 steps of 32,768 tokens (124.5M tokens, 22% of TinyStoriesV2's 562.8M), AdamW lr
1e-3 (200 warmup, cosine to 0), betas (0.9, 0.95), weight decay 0.1, float32 on MPS. It took 8,776 s
(2.44 h) in 18 chunks of about 8 minutes; validation loss 1.4028. During the run we shortened `max_iters`,
which sets the cosine schedule: 4,200 for steps 1 to 224, 4,000 for 225 to 1,314, then 3,800. The patches
to llama2.c add resumable time-limited chunks, an exact data resume, MPS fixes and atomic checkpoints
(`llama2c/PATCH_train.diff`, `llama2c/PATCH_tinystories.diff`). Recorded settings:
`configs/pro_pretrain_B.json`.

## 5. TONER-1 Pro: printer fine-tune

```sh
python -I training/scripts/pro_ft_pretokenize.py
python -I -u training/scripts/pro_ft_train.py --out_dir=$TONER_TRAIN/ckpt/pro_ft/run_ft2_tag --lr=1e-4 \
    --min_lr=5e-6 --warmup=50 --max_iters=750 --eval_interval=50 --max_wall_seconds=470 --replay_bos=2
#   repeat the same command until it reaches step 750 (it resumes from ckpt.pt)
python -I training/scripts/pro_ft_finalize.py $TONER_TRAIN/ckpt/pro_ft/run_ft2_tag   # -> finetune.pt, finetune_v0.bin
python3 tools/l2c_safetensors.py to-safetensors $TONER_TRAIN/ckpt/pro/finetune_v0.bin weights/TONER-1-Pro "TONER-1 Pro" "tok4096: ..."
```

750 steps of 16,384 tokens, half printer stories and half TinyStories replay; lr 1e-4 (50 warmup, cosine to
5e-6), weight decay 0.1; 1,467 s; about 4.7 passes over the printer stories. In the replay rows the stories
are separated by token 2 (`</s>`) instead of BOS, so BOS, where every print job starts, comes to mean "a
printer story". The first chunk (steps 1 to 200) ran with `--max_iters=900`, which sets the cosine schedule;
the later chunks used 750. Recorded settings: `configs/pro_ft_run2.json`. `pro_ft_pretokenize.py`
reproduces exactly (`train.bin` 323749b6defaa919dea0cff14917d8a2, `val.bin` de31184f47b01745ce76252fe73344aa),
and `pro_ft_finalize.py` writes `finetune_v0.bin` MD5 6b50799b3a713439d6d58d1eaf507137.

| | Pretrained | Fine-tuned (fp32) | Fine-tuned (int8) |
|---|---|---|---|
| Printer validation loss | 2.9523 | 1.9542 | 1.9543 |
| 50 samples: any printer word | | 100% | 98% |
| 50 samples: says "printer" | | 96% | 94% |
| 50 samples: ended on its own | | 90% | 86% |

TinyStories validation loss rose by 8.2% in the standard format and 3.0% in the replay format. Evaluate
with `python -I training/scripts/pro_ft_eval.py CKPT.pt OUT_PREFIX --tag=2`.

## 6. Into the print jobs

```sh
python3 tools/l2c_safetensors.py to-bin weights/TONER-1 build/toner1.bin
python3 tools/export.py build/toner1.bin checkpoints/tok512.bin build/weights.ps          # MD5 838ba89393e424c3fc7cff15e9d28f16
python3 tools/l2c_safetensors.py to-bin weights/TONER-1-Pro build/toner1_pro.bin
python3 tools/export.py build/toner1_pro.bin training/tokenizer/tok4096.bin build/weights_pro.ps   # MD5 70b3c06decb0c88672123294f94c2edb
sh tools/build.sh
```

That is what `sh tools/rebuild.sh` runs. `model.safetensors` stores the float32 weights byte for byte as
in the `.bin`; `to-bin` recomputes the RoPE tables, which can differ from PyTorch's in the last bit, but
`export.py` reads only the weights.

## Licenses

- Our code here, the TONER-1 fine-tune, and TONER-1 Pro's weights and tokenizer: MIT (`LICENSE` at the repo root).
- `llama2c/`: Andrej Karpathy's llama2.c, MIT (`llama2c/LICENSE`).
- The printer stories in `data/`: written by Xing4.0-29B-A4B (Apache-2.0; no restriction on its outputs),
  filtered by us, released under Apache-2.0.
- TinyStories (Ronen Eldan and Yuanzhi Li, 2023), CDLA-Sharing-1.0: downloaded, not committed. It is the
  base model's training data, Pro's pretraining data and the replay data of both fine-tunes.

No Llama 2 materials (weights, tokenizer or code under Meta's license) are used anywhere in this folder.
