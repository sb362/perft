
# Fast perft calculator for chess

- https://www.chessprogramming.org/Perft
- https://www.chessprogramming.org/Perft_Results

## Usage
```
  -f, --fen arg     FEN string
  -m, --moves arg   Comma-separated list of moves in UCI form to apply to the
                    root position
  -d, --depth arg   Depth
  -u, --upto        Calculate for depths 1...n
  -b, --bench       Benchmark mode
  -v, --verify arg  Compare perft results to another UCI engine
      --divide      Print move counts for each root move
  -c, --compiler    Show compiler info

Predefined FENs:
 startpos   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -
 kiwipete   r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -
 pins       8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
 cpw4       r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -
 cpw5       rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -
 cpw6       r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -
 promotions n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - -
```

Example:
```
./perft -f kiwipete -d 5 -u
...
Depth  Nodes        Time (ms)    Nodes/sec
1      48           0            6857143
2      2039         0            101950000
3      97862        0            607838509
4      4085603      4            921426026
5      193690690    196          983910687
```

Single-threaded only (for now).

## Speeds

Built w/ profile-guided optimisation \
CPU: i7-6700k (4.2 GHz) \
Compiler: GCC 9.3 \
OS: Linux \
BMI2, LSB, POPCNT enabled \
PEXT bitboards

| Name       | Depth | Nodes       | Time (ms) | Nodes/sec |
|------------|-------|-------------|-----------|-----------|
| startpos   | 7     | 3195901860  | 5554      | 575367201 |
| kiwipete   | 6     | 8031647685  | 8653      | 928149898 |
| pins       | 8     | 3009794393  | 7215      | 417147513 |
| cpw4       | 6     | 706045033   | 774       | 911751721 |
| cpw5       | 6     | 3048196529  | 4144      | 735564933 |
| cpw6       | 6     | 6923051137  | 7423      | 932543275 |
| promotions | 6     | 71179139    | 144       | 493716716 |
| total/avg  | -     | 24985815776 | 33907     | 736892552 |

## Building
Run:
- `./build.sh` for a debug build.
- `./build-release.sh` for a release build (-O3).
- `./build-pgo.sh` for a PGO build (-fprofile-generate/use).

## Dependencies
Uses [cxxopts](https://github.com/jarro2783/cxxopts) (cxxopts.hh) and [fmtlib](https://github.com/fmtlib/fmt) (fmt/, submodule).
