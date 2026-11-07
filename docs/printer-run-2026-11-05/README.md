# Printer run, 2026-11-05

On 2026-11-05 we sent two of the print jobs in this repo to an HP Color LaserJet Pro MFP M281fdw. The printer
reports its product name as "HP Color LaserJet MFP M281fdw" and runs PostScript version 3017, revision 1.
Both pages came out right. This folder holds what the printer sent back over the network while the jobs ran.

## What was sent, and how

| Job | Bytes | md5 |
|---|---|---|
| [`print-jobs/TONER-1_print-job.ps`](../../print-jobs/TONER-1_print-job.ps) | 595,372 | eed4ba82be558bf48e2cd62ae381c7ca |
| [`print-jobs/TONER-1-V_print-job.ps`](../../print-jobs/TONER-1-V_print-job.ps) | 859,530 | e4066228ac816e4c480c7907031d0141 |

- These are the files in this repo, byte for byte. Each job ran with its defaults: seed 42, temperature 0.8,
  200 tokens; TONER-1-V's doodle at temperature 0.9.
- Each file went to the printer as raw PostScript over the network, on port 9100:
  `nc -v -G 5 <printer> 9100 < job`. A small Python wrapper around `nc` saved every chunk the printer sent
  back, with the number of seconds since the send started.
- The host computes nothing for the model (zero host FLOPs). It sends the file's bytes and saves what comes
  back. The printer's PostScript interpreter does every multiply-add.
- `probe.ps` went first. It draws nothing and has no `showpage`. It asks the interpreter for its product
  name, version, revision and memory (`vmstatus`).
- TONER-1 Pro was not sent.

## What was measured

| | TONER-1 | TONER-1-V |
|---|---|---|
| Sent (EST) | 22:24:59 | 22:28:50 |
| Generation, by the printer's own clock | 200 tokens in 3:37 (0.92 tok/s) | story: 200 tokens in 3:38; doodle: 64 patch tokens in 25 s |
| Send to finished job (the printer closed the connection) | 3:39 | 4:04 |
| First line back (the transcript header) | 0.90 s | 0.96 s |
| First token back | 1.70 s | 1.79 s |
| PostScript memory at the end | not printed by this job | 1,674,376 bytes used of 109,477,988 |
| Send to the page in the output tray | not measured | not measured |

- The printer's own lines, verbatim:
  - TONER-1: `200 tokens in 217.235 s = 0.920662 tok/s [STEPS limit]`
  - TONER-1-V: `200 tokens in 217.806 s = 0.918248 tok/s [STEPS limit]`, then
    `64 patch tokens (65 forward passes) in 24.77 s = 2.58377 tok/s` and
    `VM: 1674376 bytes used of 109477988 allocated (save level 2)`
- The job times these with PostScript's `usertime`, so they are the printer's clock, not the Mac's.
- TONER-1's tokens arrived one at a time: 0.83 s per token for tokens 1 to 20, 1.08 s for 91 to 110 and
  1.35 s for 181 to 200. Later tokens take longer because each one attends over all the tokens before it.
  TONER-1-V's back-channel arrived in 7 chunks, so it gives no per-token times.
- The header reached the Mac 0.90 s after the send began. By then the printer had read and run the first
  583,376 bytes of the 595,372-byte job, which hold all of the weights.
- Nobody timed when the pages reached the output tray. There is no photo of the pages.

## Same output as Ghostscript

The story, TONER-1-V's router line, the 16 x 16 frog and its 64 patch tokens are byte-identical to
Ghostscript 10.07.1's output for the same jobs. The printer ends its lines with CRLF, so the comparison maps
CRLF to LF first. Only the timing lines and the VM line differ:

| | Printer | Ghostscript 10.07.1, Apple M4 Mac mini |
|---|---|---|
| TONER-1, 200 tokens | 217.235 s, 0.92 tok/s | 2.41 s, 82.99 tok/s |
| TONER-1-V story, 200 tokens | 217.806 s | 2.48 s |
| TONER-1-V doodle, 64 patch tokens | 24.77 s | 0.25 s |
| TONER-1-V VM line | 1,674,376 used of 109,477,988 | 3,978,524 used of 5,306,672 |

The printer generated TONER-1's tokens about 90 times more slowly than Ghostscript on the Mac. PostScript's
`rand` is implementation-defined, so another printer may tell a different story from the same seed. This one
told the same story.

## Files

| File | What it is |
|---|---|
| `TONER-1.backchannel.txt`, `TONER-1-V.backchannel.txt` | Everything the printer sent back, as received (CRLF line endings) |
| `TONER-1.timing.jsonl`, `TONER-1-V.timing.jsonl` | The same bytes, one JSON line per chunk received: `t` is seconds since the send began, `data` is the chunk |
| `TONER-1.sent_at.txt`, `TONER-1-V.sent_at.txt` | When each send began, on the Mac's clock |
| `TONER-1.result.txt`, `TONER-1-V.result.txt` | The sender's exit code, how long the connection was open, and when it closed |
| `TONER-1.status.txt`, `TONER-1-V.status.txt` | The sender's final state (`exited 0`) |
| `TONER-1.backchannel.txt.nc-stderr`, `TONER-1-V.backchannel.txt.nc-stderr` | `nc -v`'s connection message |
| `probe.ps` | The probe job |
| `probe.backchannel.txt` | The probe's answer |
| `probe.nc.txt` | `nc -v`'s connection message for the probe |

**What was scrubbed.** In the three `nc -v` connection messages (`*.nc-stderr` and `probe.nc.txt`), the
printer's network hostname is replaced with `<printer-hostname>`. Every other file is a byte-exact copy of the
log written on the day. The back-channel files keep the printer's CRLF line endings, and `.gitattributes` here
stops Git from converting them.

Only send print jobs to printers you own or have permission to use.

Not affiliated with or endorsed by Adobe, HP, Artifex, Google, or any AI lab.
