CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC

all: svm.o svm-train

svm-train: svm_train.cc svm.o utils.o
	$(CXX) $(CFLAGS) svm_train.cc svm.o utils.o -o svm-train -lm

svm.o: svm.cc svm.h
	$(CXX) $(CFLAGS) -c svm.cc

utils.o: utils.cc utils.h
	$(CXX) $(CFLAGS) -c utils.cc

clean:
	rm -f *~ *.swp svm.o utils.o svm-train