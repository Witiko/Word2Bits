CC_MIKOLOV=g++
BIT_ARRAY=./BitArray
CFLAGS=-O3 -march=native -lm -pthread -Wno-unused-result -I$(BIT_ARRAY)/ -L$(BIT_ARRAY)/

word2bits: src/word2bits.cpp
	$(CC_MIKOLOV) $(CFLAGS) $< -o word2bits

compute_accuracy: src/compute-accuracy.c
	git submodule update
	make -C $(BIT_ARRAY)
	$(CC_MIKOLOV) $(CFLAGS) $< -o compute_accuracy -lbitarr

clean:
	rm -f word2bits
	rm -f compute_accuracy
