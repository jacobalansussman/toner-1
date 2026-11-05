# PostScript matrix-vector benchmarks

These micro-benchmarks come from the development of `engine/engine2.ps` and compare ways to store and multiply weights in PostScript. Each one prints multiply-accumulates per second (MAC/s) for a 512 x 64 matrix times a vector, repeated 400 times. Run them with `gs -q -dNODISPLAY -dBATCH -dNOPAUSE bench/bench.ps`.

- `bench.ps` compares five idioms: a naive indexed `for` loop over arrays of reals, `forall` over a row, `aload` and `roll`, and two versions of an int8 string with a lookup table.
- `bench2.ps` is "compiled weights", where every row is a procedure with its weights written in as literals. It is fast, but it makes a much larger file.

One run in Ghostscript 10.07.1 on an Apple M4 gave these results:

| Idiom | MAC/s |
|---|---|
| mv1, naive `for` over arrays of reals | 35.4M |
| mv2, `forall` over arrays of reals | 32.8M |
| mv3, `aload` / `roll` | 27.9M |
| mv4, int8 string + lookup table, `for` | 22.2M |
| mv5, int8 string + lookup table, `forall` | 25.2M |
| bench2, compiled weights | 69.0M |

The shipped engine keeps the weights as int8 hex strings with one scale per row. This keeps the job at 595 KB, about a quarter of what float32 text would need. Instead of a lookup table, it subtracts `128 * sum(x)` once per matrix-vector product.
