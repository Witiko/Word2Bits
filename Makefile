CC_MIKOLOV=g++
CFLAGS=-O3 -march=native -lm -pthread -Wno-unused-result

word2bits: src/word2bits.cpp
	$(CC_MIKOLOV) $(CFLAGS) $< -o $@
compute_accuracy: src/compute-accuracy.c
	$(CC_MIKOLOV) $(CFLAGS) $< -o $@
clean:
	rm -f word2bits
	rm -f compute_accuracy
