mkdir -p bin

g++ -std=c++11 third_party/orca/orca.cpp -o bin/orca -O3
g++ -std=c++11 third_party/isomorph.cpp -O3 -o bin/subiso -fopenmp -I$HOME/boost/include
g++ -std=c++11 third_party/unique.cpp -O3 -o bin/unique -fopenmp -I$HOME/boost/include