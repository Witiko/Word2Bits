CC_MIKOLOV=g++
CFLAGS=-march=native -lm -pthread -Wno-unused-result -std=c99 -Ofast

word2bits:
	$(CC_MIKOLOV) $(CFLAGS) ./src/word2bits.cpp -o word2bits
compute_accuracy:
	$(CC_MIKOLOV) $(CFLAGS) ./src/compute-accuracy.c -o compute_accuracy
compute_accuracy_bitwise:
	$(CC_MIKOLOV) $(CFLAGS) -DBITWISE_DISTANCES ./src/compute-accuracy.c -o compute_accuracy_bitwise
clean:
	rm -f word2bits
	rm -f compute_accuracy
