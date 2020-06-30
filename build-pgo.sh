#!/bin/sh
g++ -std=c++17 -Ifmt/include -m64 -mbmi2 -msse4 -march=native -O3 -s -fprofile-generate perft.cc -o perft
./perft --bench
g++ -std=c++17 -Ifmt/include -m64 -mbmi2 -msse4 -march=native -O3 -s -fprofile-use perft.cc -o perft
