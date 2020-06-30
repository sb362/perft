#!/bin/sh
g++ -std=c++17 -Ifmt/include -m64 -mbmi2 -msse4 -march=native -O3 -s perft.cc -o perft
