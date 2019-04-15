CC_MIKOLOV=g++
CFLAGS=-march=native -lm -pthread -Wno-unused-result -std=c99 -Ofast

word2bits: src/word2bits.cpp
	$(CC_MIKOLOV) $(CFLAGS) $< -o $@

compute_accuracy: src/compute-accuracy.c
	$(CC_MIKOLOV) $(CFLAGS) $< -o $@

compute_accuracy_bitwise: src/compute-accuracy.c
	$(CC_MIKOLOV) $(CFLAGS) -DBITWISE_DISTANCES $< -o $@

clean:
	rm -f word2bits
	rm -f compute_accuracy
	rm -f compute_accuracy_bitwise
