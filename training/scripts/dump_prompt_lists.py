#!/usr/bin/env python3
"""dump_prompt_lists.py: write the prompt ingredients of gen_stories.py to training/data/prompt_lists.json.

gen_stories.py is the source; this file is a readable copy: the system prompt, the 68 printer themes, the noun,
verb, adjective and name lists, the story features, the drawable things, the sampling settings and one example
prompt (built for request 0 of the seed the bulk run used). No network, no key.
Usage: python -I dump_prompt_lists.py [OUT.json]
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TRAINING  # noqa: E402
import gen_stories as g  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TRAINING, "data", "prompt_lists.json")
seed = f"printer-{g.PROMPT_VERSION}"
d = {
    "source": "training/scripts/gen_stories.py",
    "model": g.MODEL, "model_license": g.MODEL_LICENSE, "prompt_version": g.PROMPT_VERSION,
    "sampling": {"temperature": 0.8, "top_p": 0.95, "reasoning": "disabled"},
    "system": g.SYSTEM,
    "themes": g.THEMES, "nouns": g.NOUNS, "verbs": g.VERBS, "adjectives": g.ADJS, "names": g.NAMES,
    "features": g.FEATURES, "drawables": g.DRAWABLES,
    "drawable_share": g.DRAWABLE_SHARE, "name_share": g.NAME_SHARE,
    "filters": {"words": [g.MIN_WORDS, g.MAX_WORDS], "min_printer_mentions": g.MIN_PRINTER_MENTIONS,
                "printer_words_regex": g.PRINTER_WORDS.pattern, "max_hard_words": g.MAX_HARD_WORDS,
                "simple_word_min_count_in_tinystories": g.VOCAB_MIN_COUNT},
    "example_prompt": {"seed": seed, "request": 0, "stories": 12,
                       "user": g.build_prompt(g.make_specs(random.Random(f"{seed}-0"), 12, {}))},
}
with open(out, "w") as f:
    json.dump(d, f, indent=1, ensure_ascii=True)
    f.write("\n")
print(f"{out}: {len(g.THEMES)} themes, {len(g.NOUNS)} nouns, {len(g.VERBS)} verbs, {len(g.ADJS)} adjectives, "
      f"{len(g.NAMES)} names, {len(g.FEATURES)} features")
