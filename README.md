# TONER-1

**We're proudly pushing forward the PostScript Pareto Frontier.**
TONER-1 is state of the art among language models that run entirely inside a print job. That category has very few entrants.

That's the parody. Here is the real part. TONER-1 is a small language model whose weights, forward pass, sampler and page layout are all in one PostScript file. Send that file to a PostScript printer and the printer's own processor writes a short children's story about printers, one token at a time, then typesets it and prints the page. The computer that sends the file does no arithmetic at all (0 FLOPs).

- **TONER-1** is Andrej Karpathy's llama2.c `stories260K` (260,032 parameters), fine-tuned by us on 8,757 printer stories, stored as int8 with one scale per row, plus a forward pass written in PostScript ([`engine/engine2.ps`](engine/engine2.ps)). The whole model is one 595,372-byte print job: [`print-jobs/TONER-1_print-job.ps`](print-jobs/TONER-1_print-job.ps).
- **TONER-1-V** adds a 103,536-parameter doodle model trained on Google's Quick, Draw! drawings. Its print job, [`print-jobs/TONER-1-V_print-job.ps`](print-jobs/TONER-1-V_print-job.ps) (859,530 bytes), writes a TONER-1 story and then draws a 16x16 one-bit doodle of the first drawable noun in it. The two models together have 363,568 parameters.
- **TONER-1 Pro** is our own model: a 4,096-token tokenizer and 12,194,688 parameters, trained from scratch by us on TinyStories and then fine-tuned on the same printer stories. It runs on the same engine, in one 24.9 MB print job: [`print-jobs/TONER-1-Pro_print-job.ps`](print-jobs/TONER-1-Pro_print-job.ps).

**Status**

<!-- PRINT_DAY:status -->
Printed on an HP Color LaserJet Pro MFP M281fdw on 2026-11-05: 200 tokens in 3:37 by the printer's own clock (0.92 tok/s), 3:39 from send to finished job.
<!-- /PRINT_DAY:status -->

<!-- PRINT_DAY:proof -->

**A language model that ran on an actual printer: one print job, zero host FLOPs.**

What the printer itself sent back over the network while [`print-jobs/TONER-1_print-job.ps`](print-jobs/TONER-1_print-job.ps) ran: its own output and its own timer. These are the first and last lines; the story in between is left out here.

```text
TONER-1 | 260032 params | int8 | PostScript-native inference
200 tokens in 217.235 s = 0.920662 tok/s [STEPS limit]
```

The TONER-1-V job printed the same day, doodle included: the story in 3:38 and the doodle in 25 s by the printer's own clock, 4:04 from send to finished job.

Full logs, with every byte the printer sent back and when it arrived: [`docs/printer-run-2026-11-05/`](docs/printer-run-2026-11-05/).

<!-- /PRINT_DAY:proof -->

**Video:** files in [`docs/media/`](docs/media/). The [MP4 version](docs/media/TONER-1_readme_loop.mp4) has the same loop as the image below.

[![TONER-1 in a short loop](docs/media/TONER-1_readme_loop.webp)](docs/media/TONER-1_readme_loop.mp4)

## Quickstart

**Preview with Ghostscript.** No printer is needed. Each run takes a few seconds, and any seed works.

```sh
gs -dSEED=7 -sDEVICE=pdfwrite -o page.pdf print-jobs/TONER-1_print-job.ps
gs -dSEED=7 -sDEVICE=pdfwrite -o page-v.pdf print-jobs/TONER-1-V_print-job.ps     # story + doodle
gs -dSEED=7 -sDEVICE=pdfwrite -o page-pro.pdf print-jobs/TONER-1-Pro_print-job.ps # about 70 s
```

As it generates, the story also streams to the terminal. You can override these settings:

| Setting | Default | What it does |
|---|---|---|
| `-dSEED=` | 42 | Seed for the sampler |
| `-dTEMP=` | 0.8 | Sampling temperature; 0 means always pick the most likely token |
| `-dSTEPS=` | 200 | Maximum number of tokens |
| `-dDTEMP=` | 0.9 | Temperature for the doodle model (TONER-1-V only) |

**Print it.** Only do this on a PostScript printer that you own or have permission to use. Send the file raw, for example:

```sh
nc <printer-ip> 9100 < print-jobs/TONER-1_print-job.ps
lpr -P <queue> -o raw print-jobs/TONER-1_print-job.ps
```

A printer has no command line, so to change the seed you edit the `42` in `/SEED 42 def` inside the file. The job needs a PostScript interpreter, Level 2 or later. A printer that only speaks PCL or GDI may print the file as pages of text instead. On an HP Color LaserJet Pro MFP M281fdw, TONER-1 wrote 200 tokens in 3:37 by the printer's own clock (0.92 tok/s), and took 3:39 from send to finished job. The TONER-1-V job took 4:04 from send to finished job. Other printers have not been tried, and a long job can hit a printer's job timeout or run out of memory. TONER-1 Pro is about 28 times slower and needs about 37 MB of PostScript memory in Ghostscript. It has not been run on a printer; on one it is estimated at 20 minutes to 5 hours a page (about 1.7 hours scaled from TONER-1's speed on the HP, also an estimate), if it fits at all. PostScript's `rand` is implementation-defined, so another interpreter may tell a different story from the same seed. At seed 42, the HP's story was byte-identical to Ghostscript's, apart from the printer's CRLF line endings.

## How it works

Everything below happens inside the print job. [`docs/TONER-1/01_explainer.pdf`](docs/TONER-1/01_explainer.pdf) walks through it on two pages, quoting the actual code.

- **Weights as PostScript strings.** [`tools/export.py`](tools/export.py) reads a llama2.c checkpoint and writes every matrix as `[ [row scales] [<hex> <hex> ...] ]`. Each row is a hex string of int8 values offset by 128, with one float scale per row. Inside PostScript, `string i get` returns a byte as an integer. The matrix-vector product `mv` accumulates `byte * x` along the row, subtracts `128 * sum(x)` once, and multiplies by the row's scale. The embedding matrix is shared with the output head. The vocabulary is a list of hex strings of raw bytes.
- **The forward pass** is about 100 lines and follows llama2.c's `run.c` step by step: RMSNorm, the Q/K/V projections, RoPE, causal attention over a KV cache, the output projection, a SwiGLU feed-forward block, and residual connections.
- **RoPE uses degrees.** PostScript's `sin` and `cos` take degrees, so every rotary angle is multiplied by 57.29577951.
- **Grouped-query attention.** `stories260K` has 8 query heads that share 4 key/value heads. Query head `h` reads KV head `h idiv 2`. TONER-1 Pro has 6 heads and 6 key/value heads, and runs through the same code.
- **SwiGLU:** `silu(W1 x) * (W3 x)`, then `W2`. `exp` arguments are clamped at -80 to stay within PostScript's real range.
- **Sampling.** The job takes a softmax at temperature `TEMP`, draws one number from `rand` (seeded with `srand SEED`, after discarding 4 draws), and walks the running sum until it passes that number. At `TEMP 0` it is greedy. Generation starts at the BOS token, with no prompt, and stops at an end-of-story token (BOS or EOS) or after `STEPS` tokens.
- **Typesetting.** The same job decodes the token bytes, maps a few UTF-8 punctuation marks to standard glyphs, and sets smart quotes. It then picks the largest Times size that fits the column and word-wraps the story, avoiding one-word last lines. It also draws the token stream, a "how this page was made" band, and a run card with the number of tokens, seconds and tok/s, measured with `usertime`. The footer names the model and the interpreter that ran the job.
- **TONER-1-V** ([`engine/doodle_engine.in.ps`](engine/doodle_engine.in.ps), [`engine/story_page.ps`](engine/story_page.ps)). The doodle model reuses engine2.ps's forward pass byte for byte: [`tools/build.sh`](tools/build.sh) splices it in and checks that it is identical. That code runs inside its own dictionary, so the story model's weights are never touched. A keyword router picks the first word in the story that names one of 18 classes (cat, dog, bird, fish, duck, rabbit, frog, teddy bear, sun, tree, flower, house, car, apple, cake, soccer ball, light bulb and calculator, with plurals and a few synonyms). If no word matches, the class is `seed mod 18`. The model then samples 64 tokens restricted to the patch tokens. Each token is one 2x2 patch of a 16x16 one-bit image. The job draws the image with `imagemask`.
- **Operators.** All three jobs use only PostScript Level 1 operators plus `rectstroke` (Level 2), and declare `%%LanguageLevel: 2`. To check this, `python3 -I tools/names_used.py print-jobs/*.ps` lists every name a job needs from the interpreter.

## Who trained what

| Model | Weights | Tokenizer | Training data |
|---|---|---|---|
| TONER-1 | Karpathy's `stories260K` ([tinyllamas](https://huggingface.co/karpathy/tinyllamas), trained by him on TinyStories), fine-tuned by us for 3,000 steps (about 11 minutes on an M4) | Karpathy's `tok512`, unchanged | 8,757 printer stories, mixed with TinyStories replay |
| TONER-1 Pro | Ours, trained from scratch: 2.4 h of pretraining on TinyStories, then 25 minutes of fine-tuning, on an M4 | Ours: 4,096-token SentencePiece BPE | TinyStories (124.5M tokens), then the printer stories with TinyStories replay |
| TONER-1-V doodle model | Ours, trained from scratch (about 12 minutes on an M4) | 36 tokens: 16 patch values, 18 classes, BOS, EOS | 171,000 Quick, Draw! drawings at 16x16, one bit |

**The printer stories** (9,218 in all: 8,757 for training, 461 held out) were written for this project by Xing4.0-29B-A4B, an Apache-2.0 model by China Telecom's TeleAI, through OpenRouter's free tier, for $0. They follow the TinyStories recipe: simple words, a printer theme for each story (68 themes, from "a paper jam" to "a printer that prints a bedtime story"), plus required words and story features. They are in [`training/data/`](training/data/). [`training/README.md`](training/README.md) has the generator, the filters, every training command and the settings of the shipped runs. The shipped weights are in [`weights/`](weights/) as float32 safetensors.

## Results

The Ghostscript measurements are from Ghostscript 10.07.1 on an Apple M4 Mac. The printer row is one run each of the TONER-1 and TONER-1-V jobs on an HP Color LaserJet Pro MFP M281fdw on 2026-11-05 (TONER-1 Pro was not sent): the token and doodle times are the printer's own clock, and the time from send to finished job was measured on the Mac that sent the job. The logs, with everything the printer sent back, are in [`docs/printer-run-2026-11-05/`](docs/printer-run-2026-11-05/).

| | TONER-1 | TONER-1-V | TONER-1 Pro |
|---|---|---|---|
| Model | `stories260K`, printer fine-tune: 5 layers, dim 64, 8 heads / 4 KV heads, 512-token vocabulary | doodle model, 4 layers, dim 48, 6 heads / 3 KV heads, hidden 128 | ours: 6 layers, dim 384, 6 heads / 6 KV heads, hidden 1024, 4,096-token vocabulary |
| Parameters | 260,032 | 103,536 (363,568 with the story model) | 12,194,688 |
| Print job | 595,372 bytes | 859,530 bytes | 24,910,286 bytes |
| Speed in Ghostscript | about 80 tok/s; under 4 s per page | about 0.3 s per doodle; about 2.8 s per illustrated page | about 2.8 tok/s; about 70 s per page |
| Speed on the HP printer | 200 tokens in 3:37 (0.92 tok/s); 3:39 from send to finished job | story in 3:38, doodle in 25 s (2.58 tok/s); 4:04 from send to finished job | not run on a printer; estimated 20 min to 5 h per page |

**Did the fine-tune make it write about printers?** Yes. Sampled from the int8 weights the way the print job samples (from BOS, temperature 0.8, up to 200 tokens), 87.3% of 300 TONER-1 stories use a printer word and 71.7% say "printer", against 3.7% and 0% for the original `stories260K`. Loss on 461 held-out printer stories fell from 1.83 to 1.23, while TinyStories validation loss rose 3.9% (1.341 to 1.393). TONER-1 Pro uses a printer word in 98% of 50 int8 samples and says "printer" in 94%.

**Are the stories any good?** One AI rater scored 50 shuffled Ghostscript transcripts blind, on 1-to-5 scales:

| Model | Stories | Coherence | Language | Fun | Printer-themed |
|---|---|---|---|---|---|
| Original `stories260K` | 20 | 1.90 | 3.65 | 1.90 | 0 of 20 |
| TONER-1 | 20 | 1.70 | 3.30 | 2.25 | 9 of 20 |
| TONER-1 Pro | 10 | 3.6 | 4.8 | 3.6 | 10 of 10 |

The fine-tune trades a little coherence (not significant, p = 0.37) for printer content and fun (p = 0.04). This is one AI rater on 50 stories, not a human study.

**Does the PostScript compute the right thing?** Yes, to within floating-point rounding. Teacher-forced on 96 tokens of a printer story and of a TinyStories story, TONER-1's logits in Ghostscript are within 7.0e-5 of an int8 float64 reference, and the most likely token agrees at 96 of 96 positions. For the doodle model, on three reference sequences, the logits match a float64 Python reference within 4.2e-5, and the most likely next patch agrees at all 66 of 66 positions on each sequence ([`pixelart/check/teacher_force_results.json`](pixelart/check/teacher_force_results.json)).

**Can you tell what the doodles are?** In a blind test, 36 doodles generated by the print job (2 per class, shuffled, unlabeled) were shown to an AI model that was not told the labels. It named **24 of 36** correctly. Picking at random would get about 2 right. Objects did better than animals: 17 of 20 objects against 7 of 16 animals. It named none of the dogs, rabbits or frogs (0 of 2 each). An AI model did the judging, not people, and there are only 2 doodles per class, so read the per-class numbers loosely. The details are in [`docs/TONER-1-V/06_blind_test_results.pdf`](docs/TONER-1-V/06_blind_test_results.pdf) and [`pixelart/eval_key/`](pixelart/eval_key/).

The stories and doodles in `docs/` are real model output and have not been edited.

## Benchmarks (PARODY)

> **PARODY. These are not real evaluations.** They are jokes in the shape of a model launch, and the numbers come from the model card in [`docs/TONER-1/02_model-card-and-benchmarks.pdf`](docs/TONER-1/02_model-card-and-benchmarks.pdf). Frontier models from major labs were not run on these benchmarks and are not named. "N/A" means only that they cannot execute inside a print job.

| Benchmark (parody) | "Measures" | TONER-1 | Frontier models from major labs |
|---|---|---|---|
| TPS (Tokens Per Sheet) | Output tokens per US Letter sheet. Not tokens per second | 200 | N/A |
| HelloPaperJam | Copy-room commonsense (parody of HellaSwag) | 27.9% | N/A |
| PC LOAD LETTER | Share of outputs printed on the paper size the job asked for | 100.0% | N/A |
| Jam Rate | Invented objects per 100 tokens (lower is better) | 1.5 | N/A |
| TonerBench | Dollars of toner per 1M output tokens (lower is better) | $125 | N/A |
| PostScript Pareto Frontier | Share of the model that executes inside the print job | 100% | 0% |

The [PostScript Pareto Frontier chart](docs/TONER-1-V/01_pareto_frontier.pdf) is also parody.

## Limitations

- **Printer speed comes from one printer.** TONER-1 and TONER-1-V ran once each on an HP Color LaserJet Pro MFP M281fdw. Other printers have not been tried, and TONER-1 Pro's printer speed is still an estimate.
- **It is a 260K-parameter model.** The stories are short, simple, and often don't make sense ("She loved to bring paper to a avocado"). It mostly writes about Lily, Tim and the printer.
- **There is no prompt.** Every story starts from the beginning-of-sequence token.
- **Results can depend on the interpreter.** `rand` is implementation-defined, so the same seed may tell different stories on different interpreters. On the HP, both jobs at seed 42 sent back the same story as Ghostscript 10.07.1 (and TONER-1-V the same frog), byte for byte apart from the printer's CRLF line endings. The pages in `docs/` come from Ghostscript 10.07.1. The story pages' footers, which the jobs themselves write, still say that the same seed tells a different story on another interpreter. That sentence predates the HP run, and it changes at the next rebuild of the jobs.
- **Some characters are dropped.** Non-ASCII characters are left off the page, apart from a few punctuation marks. They still appear in the interpreter's stdout.
- **TONER-1-V is crude.** It draws 16x16 one-bit doodles in 18 classes, and its animals are hard to recognize. The router is a keyword match, not a model, and it has no keyword for "calculator".
- **TONER-1 Pro has never run on a printer.** "20 minutes to 5 hours" is an estimate, and a 24.9 MB job that needs about 37 MB of PostScript memory may not fit in a printer at all. The HP that ran TONER-1 reports about 110 MB of PostScript memory; Pro was not tried on it. Scaled from TONER-1's 0.92 tok/s there, Pro would take about 1.7 hours a page (an estimate). Its terminal transcript still begins "TONER-1 |"; the page footer names it correctly.
- **Printer compatibility.** The jobs need a PostScript interpreter, Level 2 or later. Printers that only speak PCL or GDI won't run them, and long jobs can hit a job timeout or memory limit.

## Reproduce

Requirements: `sh`, `curl`, Python 3 (standard library only), and Ghostscript for previews. This rebuilds all three print jobs from a fresh clone and checks them byte for byte:

```sh
sh tools/rebuild.sh
```

It runs these steps, writing into `checkpoints/` and `build/`, which git ignores:

```sh
sh tools/fetch_checkpoints.sh      # tok512.bin from huggingface.co/karpathy/tinyllamas (SHA-256 checked)
mkdir -p build
python3 tools/l2c_safetensors.py to-bin weights/TONER-1 build/toner1.bin          # our fine-tuned stories260K
python3 tools/l2c_safetensors.py to-bin weights/TONER-1-Pro build/toner1_pro.bin  # our Pro
python3 tools/export.py build/toner1.bin checkpoints/tok512.bin build/weights.ps
python3 tools/export.py build/toner1_pro.bin training/tokenizer/tok4096.bin build/weights_pro.ps
sh tools/build.sh                  # assembles build/*.ps and compares them with print-jobs/
```

`tools/build.sh` should end with `byte-identical to print-jobs/...` for `TONER-1_print-job.ps`, `TONER-1-Pro_print-job.ps` and `TONER-1-V_print-job.ps`. The inputs are the committed float32 weights in [`weights/`](weights/) (`model.safetensors` and `config.json` for each model), the committed Pro tokenizer in [`training/tokenizer/`](training/tokenizer/), and Karpathy's `tok512.bin`. The TONER-1-V job also reads the doodle weights from [`engine/doodle_weights.ps`](engine/doodle_weights.ps), which [`pixelart/src/export.py`](pixelart/src/export.py) exported from [`pixelart/ckpt/d48L4_16k.pt`](pixelart/ckpt/).

**Training** the printer stories, the TONER-1 fine-tune and TONER-1 Pro is documented in [`training/README.md`](training/README.md). It needs PyTorch and SentencePiece, and the TinyStories download (2.2 GB).

**Pages.** `sh tools/make_pages.sh` re-renders the PDFs in `docs/` into `build/pages/`, one Ghostscript run each (`PRO=1` adds the Pro page). `sh pixelart/results/build_results.sh` rebuilds the blind-test results page. `sh tools/render_check.sh 7 0.8 seed7` makes a quick PNG preview.

**The TONER-1-V model** (data, training, int8 export and the correctness checks) is documented in [`pixelart/README.md`](pixelart/README.md). The Quick, Draw! data is downloaded by a script, not stored in this repo.

## Repo layout

| Path | What's in it |
|---|---|
| [`print-jobs/`](print-jobs/) | The three print jobs. Each one is the whole model, ready to send to a printer |
| [`weights/`](weights/) | The float32 weights of TONER-1 (our fine-tune) and TONER-1 Pro, as safetensors with a `config.json` each |
| [`engine/`](engine/) | The PostScript sources: `engine2.ps` (forward pass, sampler, page), the TONER-1-V doodle engine template, page programs, and `doodle_weights.ps` |
| [`tools/`](tools/) | `rebuild.sh`, `fetch_checkpoints.sh`, `l2c_safetensors.py`, `export.py`, `build.sh`, `make_pages.sh`, `render_check.sh`, the engine splice (`mk_engine.sh`), `wrapws.py` and `names_used.py` |
| [`training/`](training/) | The printer-story generator and dataset, the TONER-1 fine-tune, the Pro tokenizer and the Pro training code |
| [`pixelart/`](pixelart/) | The TONER-1-V training pipeline, its checkpoints, the correctness checks, and the blind test with its answer key |
| [`bench/`](bench/) | PostScript speed benchmarks for different matrix-vector idioms |
| [`docs/`](docs/) | Release PDFs, with their PostScript sources in `docs/*/src/`. Video files go in `docs/media/`, and the printer's output from 2026-11-05 is in `docs/printer-run-2026-11-05/` |

### Documents

| File | Contents |
|---|---|
| [`docs/TONER-1/TONER-1_booklet.pdf`](docs/TONER-1/TONER-1_booklet.pdf) | All 10 TONER-1 pages in one PDF |
| [`docs/TONER-1/01_explainer.pdf`](docs/TONER-1/01_explainer.pdf) | How a print job runs a language model (2 pages) |
| [`docs/TONER-1/02_model-card-and-benchmarks.pdf`](docs/TONER-1/02_model-card-and-benchmarks.pdf) | Model card and parody benchmarks (2 pages) |
| `docs/TONER-1/03`-`07_story_*.pdf` | Story pages the TONER-1 job made in Ghostscript 10.07.1. The seed and temperature are in each name; `t0` means greedy |
| [`docs/TONER-1/08_story_pro-12M_seed42_t0.8.pdf`](docs/TONER-1/08_story_pro-12M_seed42_t0.8.pdf) | One story from TONER-1 Pro, made by its job in Ghostscript |
| [`docs/TONER-1-V/TONER-1-V_booklet.pdf`](docs/TONER-1-V/TONER-1-V_booklet.pdf) | All 6 TONER-1-V pages in one PDF |
| [`docs/TONER-1-V/01_pareto_frontier.pdf`](docs/TONER-1-V/01_pareto_frontier.pdf) | The PostScript Pareto Frontier (parody chart) |
| [`docs/TONER-1-V/02_sample_sheet.pdf`](docs/TONER-1-V/02_sample_sheet.pdf) | 72 doodles: 18 classes x 4, seeds 1 to 72, not edited |
| `docs/TONER-1-V/03`-`05_story_illustrated_seed*.pdf` | The TONER-1-V job's own output for seeds 7 (a dog), 4 (the sun) and 6 (a fish) |
| [`docs/TONER-1-V/06_blind_test_results.pdf`](docs/TONER-1-V/06_blind_test_results.pdf) | Blind identification test: 24 of 36 |

All of these PDFs are Ghostscript 10.07.1 renders, not scans of the pages the HP printed. The explainer, the model card and the TONER-1 booklet were updated after the printer run of 2026-11-05.

The TONER-1 pages call the print job `toner1.ps`, which was its working name. It is the same file as `print-jobs/TONER-1_print-job.ps`.

## What changed

- **TONER-1 now writes about printers.** It is `stories260K` fine-tuned by us on 8,757 printer stories. The job grew from 595,177 to 595,372 bytes; the architecture, the parameter count and the engine are the same. The engine also stops on EOS now, and a job can name its model in the footer.
- **TONER-1 Pro is ours.** An earlier, unreleased Pro draft used Karpathy's `stories15M`, whose 32,000-token vocabulary comes from Meta's Llama 2 tokenizer. It is gone. The new Pro has our own tokenizer and weights, trained from scratch, and its print job is in this repo.
- **New pages.** Every story page in `docs/` was re-rendered from the new jobs, with the same seeds. The illustrated TONER-1-V pages are now seeds 7, 4 and 6, because the fine-tuned stories for seeds 14 and 25 drew less readable doodles. The sample sheet and the blind test use the doodle model only and are unchanged.

## Prior art

- **EXO Labs, llama98.c**: the same `stories260K` base model, running on a Windows 98 Pentium II at 39.31 tok/s (https://blog.exolabs.net/day-4/).
- **Nicolas Seriot, PSChess**: a chess program written in PostScript that runs on printers.
- **int-llm-postscript** (https://github.com/nmicic/int-llm-postscript, Aug 2026): a 14K-parameter microGPT in PostScript, run under Ghostscript.
- **llama.ttf**: a language model inside a font file.

## Credits and licenses

- Model architecture, inference reference, training code and the `stories260K` base weights: Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [tinyllamas](https://huggingface.co/karpathy/tinyllamas), MIT License.
- TinyStories by Ronen Eldan and Yuanzhi Li (2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)), CDLA-Sharing-1.0: the base model's training data, TONER-1 Pro's pretraining data, and replay data in both fine-tunes.
- The printer stories: written by Xing4.0-29B-A4B (China Telecom's TeleAI, Apache-2.0) through OpenRouter's free tier, and filtered by us. They are released under Apache-2.0.
- Training data of the TONER-1-V doodle model: Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), CC BY 4.0. 171,000 of its drawings were used, rasterized to 16x16 one-bit images.
- This repo's own code, including the PostScript engines and the Python tools, is under the MIT License ([`LICENSE`](LICENSE)), and so are the TONER-1 fine-tune and TONER-1 Pro's weights and tokenizer. The TONER-1-V weights are released under CC BY 4.0, with attribution to Google's Quick, Draw! dataset. Each print job bundles model weights under the terms listed in [`THIRD_PARTY.md`](THIRD_PARTY.md). No Llama 2 materials are in this repo.

Made by Jacob Sussman. TONER-1 is a working print job and a parody of frontier-model launches.

Not affiliated with or endorsed by Adobe, HP, Artifex, Google, or any AI lab.
