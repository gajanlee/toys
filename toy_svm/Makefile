CXX ?= g++
CFLAGS = -Wall -Wconversion -03 -fPIC

all: svm-train

svm-train: svm-train.cc svm.o
	$(CXX) $(CFLAGS) svm-train.cc svm.o -o svm-train -lm

svm.o: svm.cc svm.h
	$(CXX) $(CFLAGS) -c svm.cc

clean:
	rm -f *~ *.swp svm.o svm-train