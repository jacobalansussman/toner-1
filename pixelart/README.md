# TONER-1-V: the doodle model

TONER-1-V is a 103,536-parameter int8 transformer that draws 16x16 one-bit doodles. It has the same architecture as llama2.c (dim 48, 4 layers, 6 query heads / 3 KV heads, hidden 128). A doodle is 64 tokens, and each token is one 2x2 patch (value `8*TL + 4*TR + 2*BL + 1*BR`). Each sequence is BOS, a class token, then the 64 patches in raster order over the 8x8 patch grid. The model has 18 classes and was trained on 171,000 Quick, Draw! drawings (Google, CC BY 4.0), with 9,000 more held out for validation. The drawings were downsampled from 28x28 and rasterized to 16x16 one-bit images.

This folder is the training pipeline and the evidence for its numbers. The PostScript side is in [`../engine/`](../engine/), and [`../tools/build.sh`](../tools/build.sh) assembles the print job from it.

Everything here was run in local Ghostscript only. No script sends anything to a printer or any other device.

## Layout

| Path | What's in it |
|---|---|
| `src/` | The pipeline. [`src/reproduce.sh`](src/reproduce.sh) runs every step in order |
| `src/vendor/model.py` | Karpathy's llama2.c `model.py`, vendored unmodified. MIT notice in `src/vendor/NOTICE.txt` |
| `ckpt/d48L4_16k.pt`, `.json` | The shipped model (435 KB) and its training log: 16,000 steps, val 0.313 bits per pixel |
| `ckpt/judge_cnn.pt`, `.json` | A small CNN trained on real 16x16 drawings, used to score generated doodles (81% on real validation images) |
| `out/` | Where `src/export.py` writes `doodle_weights.ps`, `doodle_meta.json` and `reference_logits.json`. The shipped JSONs are committed |
| `check/` | Correctness and speed checks of the PostScript engine, with their results (`*.json`) |
| `eval_key/` | The blind identification test: seeds, answer key, the evaluator's guesses, scores. Everything here reveals the answers |
| `results/` | Builds the blind-test results page (`docs/TONER-1-V/06_blind_test_results.pdf`) |

`data/` (the downloaded drawings, about 150 MB once processed) is not committed.

## Reproduce

You need Python with `torch`, `numpy` and `pillow`, plus Ghostscript. Run these from the repo root:

```sh
sh tools/rebuild.sh        # writes build/weights.ps (the TONER-1 story weights) and the print jobs
PYTHON=/path/to/venv/bin/python sh pixelart/src/reproduce.sh
```

`reproduce.sh` runs these steps:

1. `src/fetch_data.py` downloads the first 10,000 drawings of each class from `storage.googleapis.com/quickdraw_dataset`, using HTTP Range requests (about 135 MB). It parses the `.npy` headers without pickle.
2. `src/preprocess.py` makes the 16x16 one-bit images and splits off 5% for validation.
3. `src/train.py` trains the model (about 12 minutes on an M4 using MPS).
4. `src/sample.py` samples doodles and `src/classifier.py` scores them with the judge CNN.
5. `src/export.py` writes `out/doodle_weights.ps`: int8 with one scale per row, the same packing as `tools/export.py`, all inside `/DOODLE`. It also writes float64 reference logits.
6. `src/check_ps.py` runs engine2.ps's forward pass on the new weights in Ghostscript and compares the result with the reference.

A retrained checkpoint will not be bit-identical on a different machine or torch version. That's why the shipped `ckpt/d48L4_16k.pt` is included. Running only `src/export.py` on it (after steps 1 and 2) reproduces `engine/doodle_weights.ps` byte for byte. To ship new weights, copy `out/doodle_weights.ps` to `engine/` and run `sh tools/build.sh`.

## Checks

To run the checks, first run `sh tools/build.sh`, which writes `build/doodle_engine.ps`. Then copy the shipped weights to where the checks read them, `cp engine/doodle_weights.ps pixelart/out/`, or re-export them. Set `GS=` if `gs` is not on your PATH.

- **Correctness** (`check/teacher_force.py pixelart print-jobs/TONER-1_print-job.ps`). This feeds the 3 reference sequences in `out/reference_logits.json` through the PostScript forward pass and prints every logit exactly, as an integer mantissa and a power-of-two exponent. Compared with the float64 reference:
  - the largest absolute difference is 4.2e-5, with logits as large as 27.1
  - the most likely patch token agrees at 66 of 66 positions on all 3 sequences
  - the results are bit-identical when the story model runs first in the same job, and the story model's own logits are unchanged after the doodle runs
- **It is exactly float32** (`check/f32_emulate.py`). A numpy re-implementation of Ghostscript's arithmetic (float32 reals, PostScript loop order) reproduces Ghostscript's logits bit for bit, for all 3 x 66 x 36 values. So the 4.2e-5 gap is all float32 rounding.
- **The sampler, exactly** (`check/sampler_emulate.py`). This recreates doodles in Python with Ghostscript's `rand` (Park-Miller, `srand` plus 4 discarded draws). The 72 sample-sheet doodles and the 3 story-page doodles (seeds 7, 4 and 6) all match Ghostscript's, 64 of 64 patches each.
- **The sampler, statistically** (`check/bench_judge.py`). The judge CNN identifies 77.8% of 288 PostScript doodles generated at T 0.9, against 77.1% for the Python int8 sampler. `check/seed_structure.py` finds that consecutive seeds do not make near-duplicate doodles.
- **Level 2 only** (`python3 -I tools/names_used.py print-jobs/TONER-1-V_print-job.ps`). This lists every name the job takes from the interpreter. They are all Level 1 operators, plus `rectstroke` (Level 2).

## Speed (Ghostscript 10.07.1, Apple M4)

| Measurement | Doodle tokens/s | Seconds per doodle |
|---|---|---|
| 288 doodles, doodle model only (`check/bench_judge.py`) | 228 | 0.28 |
| Sample sheet, 72 doodles | 210 | 0.31 |
| Blind grid, 36 doodles | 225 | 0.28 |

The story model in the same job (the TONER-1 fine-tune) generates about 80 tok/s. A whole illustrated page takes about 2.8 s. A printer is estimated to take about 1 to 15 minutes per illustrated page, assuming the same 0.3 to 5 tok/s as TONER-1. That has not been measured.

## Blind test

The test used 36 doodles, 2 per class, shuffled, with seeds from one fixed RNG (`eval_key/make_blind.py`, not re-rolled). They were drawn by the print job in Ghostscript at T 0.9. An AI model that was not told the labels named them from the grid image alone. It got **24 of 36** right (Wilson 95% interval 50% to 80%), where picking at random would get about 2. Objects did better than animals: 17 of 20 against 7 of 16. It named none of the dogs, rabbits or frogs (0 of 2 each). The evaluator was an AI model, not people, and with only 2 doodles per class the per-class numbers are very noisy. To re-score the test and rebuild the page, run `sh pixelart/results/build_results.sh`.

## Pages and seeds

The illustrated story pages use seeds 7 (a dog, word 23 of 89), 4 (the sun) and 6 (a fish), with story TEMP 0.8, STEPS 200 and doodle DTEMP 0.9. They tell TONER-1's printer stories. To pick them, the combined job was run for 24 seeds (1 to 21 and 23 to 25). Nineteen of those stories named a drawable noun, and five fell back to `seed mod 18`. We chose three of the nineteen, each from a different class and each legible. Seed 2 (cake) was passed over because its story calls Lily "a calculator", and the router has no keyword for that. Before the printer fine-tune the pages were seeds 7 (a house), 14 (a tree) and 25 (a car); with the fine-tune, seeds 14 and 25 draw a hard-to-read cat and a second dog, so they were replaced. The sample sheet uses seeds 1 to 72 in order, with nothing re-rolled, and it runs the doodle model only, so it did not change. At the default seed 42 in Ghostscript, the story names nothing drawable, so the job falls back to class `42 mod 18`, which is a frog.

## Known limitations

- `rand` is implementation-defined, so on a printer the same seed tells a different story and draws a different doodle.
- Animals (dog, rabbit, frog, bird, duck) often come out as scribbles. That is a limit of the 16x16 model, not of the PostScript port.
- `usertime` ticks in 10 ms steps, so the doodle tok/s printed on a single story page is only good to about ±4%.
