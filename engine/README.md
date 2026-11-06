# PostScript sources

`tools/build.sh` assembles these files into the print jobs. Each job is a DSC header, followed by the weights, a `/MODELNAME` line for the page footer, and the engine.

| File | Used in | What it is |
|---|---|---|
| `engine2.ps` | all three jobs | TONER-1 and TONER-1 Pro: run parameters, forward pass (RMSNorm, int8 matvec, RoPE, GQA attention, SwiGLU), sampler, and page layout |
| `doodle_engine.in.ps` | TONER-1-V | Doodle engine template. `tools/mk_engine.sh` splices engine2.ps's forward pass in at `%%ENGINE2-FORWARD%%`, byte for byte |
| `doodle_weights.ps` | TONER-1-V | TONER-1-V int8 weights, all inside `/DOODLE` (written by `pixelart/src/export.py`; CC BY 4.0) |
| `story_page.ps` | TONER-1-V | Keyword router, doodle drawing, and the illustrated page |
| `house.ps` | sample sheet, blind grid, results page | House style for the TONER-1-V handout pages |
| `sample_sheet_page.ps`, `blind_grid_page.ps` | sample sheet, blind grid | Page programs for those two jobs |

The story weights (`build/weights.ps` for TONER-1 and TONER-1-V, `build/weights_pro.ps` for TONER-1 Pro) are not committed as PostScript. `sh tools/rebuild.sh` regenerates them exactly: it converts the committed float32 weights in `weights/` back to llama2.c checkpoints with `tools/l2c_safetensors.py`, then runs `tools/export.py` with Karpathy's `tok512.bin` (TONER-1) or our `training/tokenizer/tok4096.bin` (Pro).
