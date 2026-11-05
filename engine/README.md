# PostScript sources

`tools/build.sh` assembles these files into the print jobs. Each job is a DSC header, followed by the weights, followed by the engine.

| File | Used in | What it is |
|---|---|---|
| `engine2.ps` | both jobs | TONER-1: run parameters, forward pass (RMSNorm, int8 matvec, RoPE, GQA attention, SwiGLU), sampler, and page layout |
| `doodle_engine.in.ps` | TONER-1-V | Doodle engine template. `tools/mk_engine.sh` splices engine2.ps's forward pass in at `%%ENGINE2-FORWARD%%`, byte for byte |
| `doodle_weights.ps` | TONER-1-V | TONER-1-V int8 weights, all inside `/DOODLE` (written by `pixelart/src/export.py`; CC BY 4.0) |
| `story_page.ps` | TONER-1-V | Keyword router, doodle drawing, and the illustrated page |
| `house.ps` | sample sheet, blind grid, results page | House style for the TONER-1-V handout pages |
| `sample_sheet_page.ps`, `blind_grid_page.ps` | sample sheet, blind grid | Page programs for those two jobs |

The TONER-1 story weights (`build/weights.ps`) are not committed, because `tools/export.py` regenerates them exactly from Karpathy's `stories260K.bin` and `tok512.bin`.
