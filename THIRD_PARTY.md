# Third-party components and licenses

This repo's own code is under the MIT License in [`LICENSE`](LICENSE), copyright 2026 Jacob Sussman. That covers the PostScript engines in `engine/`, the Python and shell tools in `tools/`, `training/scripts/` and `pixelart/`, the benchmarks in `bench/`, and the document sources in `docs/*/src/`. The same MIT License covers the weights we trained: the TONER-1 fine-tune (`weights/TONER-1/`) and TONER-1 Pro's weights and tokenizer (`weights/TONER-1-Pro/`, `training/tokenizer/`). The components below come from other people, or are under other terms.

**No Llama 2 materials are in this repo.** No Llama 2 weights, no Llama 2 tokenizer or vocabulary, and no code under Meta's license. "Llama-2 architecture" on the pages describes the shape of the network only. llama2.c's `tokenizer.py`, which carries a Meta copyright header, is left out of `training/llama2c/`.

## What each print job bundles

Each print job contains model weights, and those weights come under the terms below.

| Print job | Bundled weights | Terms |
|---|---|---|
| `print-jobs/TONER-1_print-job.ps` | The TONER-1 fine-tune of `stories260K`, converted to int8, plus Karpathy's 512-token vocabulary (`tok512.bin`) | MIT. The base model and vocabulary are Karpathy's (llama2.c / tinyllamas, MIT); the fine-tune is ours (MIT). Training data: TinyStories (CDLA-Sharing-1.0) and the printer stories (Apache-2.0) |
| `print-jobs/TONER-1-V_print-job.ps` | The same TONER-1 weights and vocabulary, plus the TONER-1-V doodle weights | MIT for the story model, as above. CC BY 4.0 for the TONER-1-V weights, with attribution to Google's Quick, Draw! dataset |
| `print-jobs/TONER-1-Pro_print-job.ps` | TONER-1 Pro, converted to int8, plus our 4,096-token vocabulary (`tok4096.bin`) | MIT, all ours. Training data: TinyStories (CDLA-Sharing-1.0) and the printer stories (Apache-2.0) |

## llama2.c and tinyllamas (MIT)

TONER-1 starts from Andrej Karpathy's `stories260K` model and its 512-token tokenizer, from [tinyllamas](https://huggingface.co/karpathy/tinyllamas). We fine-tuned it; the tokenizer is unchanged. The forward pass in `engine/engine2.ps` follows [llama2.c](https://github.com/karpathy/llama2.c). `training/llama2c/` holds llama2.c's `model.py`, `export.py` and `configurator.py` unchanged, and its `train.py` and `tinystories.py` with our patches (`PATCH_*.diff`); TONER-1 Pro was trained with them. `tools/export.py` and `tools/l2c_safetensors.py` (ours) read and write llama2.c's checkpoint format. `pixelart/src/vendor/model.py` is llama2.c's `model.py`, copied without changes, and the TONER-1-V model and its training use it.

Karpathy's files are not committed. `tools/fetch_checkpoints.sh` downloads `tok512.bin` (and, with `--train`, `stories260K.pt`, `stories260K.bin` and `tok512.model`) from his public Hugging Face repo and checks their SHA-256.

Here is the license of llama2.c, verbatim (also in `training/llama2c/LICENSE`):

```
MIT License

Copyright (c) 2023 Andrej

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## The printer stories (Apache-2.0)

The 9,218 printer stories in `training/data/` (8,757 train, 461 validation), and the 601 pilot stories that trained the Pro tokenizer (590 of them are also among the 9,218) were written by **Xing4.0-29B-A4B**, a model by China Telecom AI Technology Co., Ltd. (TeleAI), released under the Apache License 2.0 with no extra use policy or restriction on its outputs ([its license](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/LICENSE)). We ran it through OpenRouter's free tier (`xingchen-agi/xing4.0-29b-a4b:free`, provider Phala), at a cost of $0, with prompts written by us, and then filtered the stories with our scripts. We release the stories under the Apache License 2.0, matching the generator's terms; the text is in [`training/data/LICENSE`](training/data/LICENSE). The generator wrote training stories only. It was not run on any benchmark or chart.

## TinyStories (CDLA-Sharing-1.0)

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is by Ronen Eldan and Yuanzhi Li ("TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) and is licensed under the [Community Data License Agreement, Sharing, Version 1.0](https://cdla.dev/sharing-1-0/). Karpathy trained `stories260K` on it. We used it to pretrain TONER-1 Pro and to train its tokenizer, as replay data in both fine-tunes, and for word counts that decide which simple words the story prompts may ask for. This repo contains no TinyStories text: `training/README.md` explains how to download it. The stories in `docs/` were generated by the models.

## Quick, Draw! (CC BY 4.0)

The TONER-1-V doodle model, and the judge CNN used to score it, were trained on 171,000 drawings from Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). That dataset is made available by Google, Inc. under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The changes made were: Google's 28x28 `numpy_bitmap` renderings of 18 categories were rasterized to 16x16 one-bit images and tokenized as 2x2 patches. The drawings themselves are not committed. `pixelart/src/fetch_data.py` downloads them.

The TONER-1-V weights are released under **CC BY 4.0**, with attribution to Google's Quick, Draw! dataset. They are in `engine/doodle_weights.ps`, `pixelart/ckpt/d48L4_16k.pt` and inside `print-jobs/TONER-1-V_print-job.ps`. The same terms apply to the judge CNN in `pixelart/ckpt/judge_cnn.pt`. Suggested attribution:

> TONER-1-V by Jacob Sussman, trained on the Quick, Draw! dataset by Google (CC BY 4.0), rasterized to 16x16 one-bit.

## Not affiliated

Not affiliated with or endorsed by Adobe, HP, Artifex, Google, or any AI lab. PostScript is a trademark of Adobe. Ghostscript is a product of Artifex.
