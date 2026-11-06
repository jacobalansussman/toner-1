"""paths.py: where the training scripts read and write. Every script imports it; no absolute path is baked in.

  TRAINING  the training/ folder of the repo (this file's parent's parent)
  REPO      the repo root
  TT        the work directory: $TONER_TRAIN, default training/work (git ignores it). Downloads, tokenized
            streams, checkpoints, evals and logs go here, in the layout the scripts expect:
              TT/data/tinystories/TinyStoriesV2-GPT4-{train,valid}.txt   (downloaded, see training/README.md)
              TT/data/vocab/, TT/data/ft260/, TT/data/tok4096/, TT/data/tok4096_printer/
              TT/ckpt/, TT/eval/, TT/logs/
  L2C       training/llama2c: llama2.c's model.py, export.py and configurator.py, and its train.py and
            tinystories.py with our patches (MIT)
  KARPATHY  checkpoints/ at the repo root: stories260K.pt and tok512.model (sh tools/fetch_checkpoints.sh --train)
  PDATA     the printer-story text files: $PRINTER_DATA, default training/data (committed). build_datasets.py
            writes new ones to TT/data; point PRINTER_DATA there to train on them.
  TOK4096   the TONER-1 Pro tokenizer, as a path prefix (.model, .bin): $TOK4096, default
            training/tokenizer/tok4096 (committed). pro_tokenizer.py writes a retrained one to TT/tokenizer/.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING)
TT = os.path.abspath(os.environ.get("TONER_TRAIN") or os.path.join(TRAINING, "work"))
L2C = os.path.join(TRAINING, "llama2c")
KARPATHY = os.path.join(REPO, "checkpoints")
PDATA = os.path.abspath(os.environ.get("PRINTER_DATA") or os.path.join(TRAINING, "data"))
TOK4096 = os.path.abspath(os.environ.get("TOK4096") or os.path.join(TRAINING, "tokenizer", "tok4096"))
os.environ["TONER_TRAIN"] = TT   # read by llama2c/tinystories.py and configs/pro_B.py
