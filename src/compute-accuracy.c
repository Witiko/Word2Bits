//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>

#ifndef BITWISE_DISTANCES
typedef float feature_t;
typedef float distance_t;
#else
typedef uint32_t feature_t;
typedef unsigned long long distance_t;

#define RESERVE_BITS(n) (((n)+0x1f)>>5)
#define DW_INDEX(x) ((x)>>5)
#define BIT_INDEX(x) ((x)&0x1f)
#define getbit(array, index) (((array)[DW_INDEX(index)]>>BIT_INDEX(index))&1)
#define putbit(array, index, bit) \
    ((bit)&1 ?  ((array)[DW_INDEX(index)] |= 1<<BIT_INDEX(index)) \
             :  ((array)[DW_INDEX(index)] &= ~(1<<BIT_INDEX(index))) \
             , 0 \
    )
static inline void bit_array_print(feature_t* array, size_t num_features) {
  for (size_t index = 0; index < num_features; index++) {
    printf("%d", getbit(array, index));
  }
  printf("\n");
}
static inline distance_t bit_array_hamming_distance(feature_t* x, feature_t* y, size_t size) {
  distance_t distance = 0;
  for (size_t a = 0; a < size; a++) {
    distance += __builtin_popcount(x[a] ^ y[a]);
  }
  return distance;
}
static inline void bit_array_not(feature_t* dest, feature_t* x, size_t size) {
  for (size_t a = 0; a < size; a++) {
    dest[a] = ~x[a];
  }
}
static inline void bit_array_and(feature_t* dest, feature_t* x, feature_t* y, size_t size) {
  for (size_t a = 0; a < size; a++) {
    dest[a] = x[a] & y[a];
  }
}
static inline void bit_array_or(feature_t* dest, feature_t* x, feature_t* y, size_t size) {
  for (size_t a = 0; a < size; a++) {
    dest[a] = x[a] | y[a];
  }
}
#endif

const size_t max_size = 2000;         // max length of strings
const size_t N = 1;                   // number of closest words
const size_t max_w = 50;              // max length of vocabulary entries

float quantize(float num, int bitlevel) {

  if (bitlevel == 0) {
    // Special bitlevel 0 => full precision
    return num;
  }
  
  // Extract sign
  float retval = 0;
  float sign = num < 0 ? -1 : 1;
  num *= sign;
  
  // Boundaries: 0
  if (bitlevel == 1) {
    return sign / 3;
  }
  
  // Determine boundary and discrete activation value (2 bits)
  // Boundaries: 0, .5
  if (bitlevel == 2) {
    if (num >= 0 && num <= .5) retval = .25; 
    else retval = .75;
  }

  // Determine boundary and discrete activation value (4 bits = 16 values)
  // Boundaries: 0, .1, .2, .3, .4, .5, .6, .7, .8
  //real boundaries[] = {0, .25, .5, .75, 1, 1.25, 1.5, 1.75};
  if (bitlevel >= 4) {
    int segmentation = pow(2, bitlevel-1);
    int casted = (num * segmentation) + (float).5;
    casted = casted > segmentation ? segmentation : casted;
    retval = casted / (float)segmentation;
  }

  return sign * retval;
}

int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
  distance_t dist, bestd[N];
  size_t words, size, num_features, a, b, c, d, b1, b2, b3, threshold = 0;
  clock_t time;
  int bitlevel = 0, binary = 1;
  feature_t *M;
  char *vocab;
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  if (argc < 2) {
    printf("Usage: ./compute-accuracy [-binary 0|1] <FILE> <bitlevel> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n");
    return 0;
  }
  for (a = 1, b = 0; a < argc; a++) {
    if (!strcmp(argv[a], "-binary")) {
      binary = atoi(argv[++a]);
    } else switch (b++)
    {
      case 0:
        strcpy(file_name, argv[a]);
        break;
      case 1:
        bitlevel = atoi(argv[a]);
        break;
      case 2:
        threshold = atoi(argv[a]);
        break;
    }
  }
  time = clock();
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  if (threshold) if (words > threshold) words = threshold;
  fscanf(f, "%lld", &num_features);
#ifndef BITWISE_DISTANCES
  size = num_features;
  distance_t len;
#else
  size = RESERVE_BITS(num_features);
#endif
  float feature;
  feature_t vec[size];
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (feature_t *)malloc(words * size * sizeof(feature_t));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(feature_t) / 1048576);
    return -1;
  }
  printf("Starting eval...\n");
  fflush(stdout);
  for (b = 0; b < words; b++) {
#ifdef BITWISE_DISTANCES
    for (a = 0; a < size; a++) M[a + b * size] = 0;
#endif
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
    if (binary) {
      for (a = 0; a < num_features; a++) {
        fread(&feature, sizeof(feature_t), 1, f);
#ifndef BITWISE_DISTANCES
        M[a + b * size] = feature;
#else
        putbit(M + b * size, a, feature > 0);
#endif
      }
    } else {
      for (a = 0; a < num_features; a++) {
        fscanf(f, "%f ", &feature);
#ifndef BITWISE_DISTANCES
        M[a + b * size] = feature;
#else
        putbit(M + b * size, a, feature > 0);
#endif
      }
    }
#ifndef BITWISE_DISTANCES
    for (a = 0; a < num_features; a++) M[a+b*size] = quantize(M[a+b*size], bitlevel);
    len = 0;
    for (a = 0; a < num_features; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < num_features; a++) M[a + b * size] /= len;
#endif
  }
  fclose(f);
  time = clock() - time;
  printf("Loaded input file in %f seconds\n", ((double)time) / CLOCKS_PER_SEC);
  time = clock();
  TCN = 0;
  while (1) {
    for (a = 0; a < N; a++) {
#ifndef BITWISE_DISTANCES
      bestd[a] = 0;
#else
      bestd[a] = num_features;
#endif
    }
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    scanf("%s", st1);
    for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
    if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(stdin)) {
      if (TCN == 0) TCN = 1;
      if (QID != 0) {
	printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (float)TCN * 100, CCN, TCN);
	printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (float)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
      }
      QID++;
      scanf("%s", st1);
      if (feof(stdin)) break;
      printf("%s:\n", st1);
      TCN = 0;
      CCN = 0;
      continue;
    }
    if (!strcmp(st1, "EXIT")) break;
    scanf("%s", st2);
    for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
    scanf("%s", st3);
    for (a = 0; a<strlen(st3); a++) st3[a] = toupper(st3[a]);
    scanf("%s", st4);
    for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
    b1 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;
    b2 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;
    b3 = b;
    for (a = 0; a < N; a++) {
#ifndef BITWISE_DISTANCES
      bestd[a] = 0;
#else
      bestd[a] = num_features;
#endif
    }
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;

#ifndef BITWISE_DISTANCES
    for (a = 0; a < num_features; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
#else
    bit_array_not(vec, M + b1 * size, size);
    bit_array_and(vec, M + b2 * size, vec, size);
    bit_array_or(vec, M + b3 * size, vec, size);
#endif
    
    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
#ifndef BITWISE_DISTANCES
      dist = 0;
      for (a = 0; a < num_features; a++) {
        dist += vec[a] * M[a + c * size];
      }
#else
      dist = bit_array_hamming_distance(vec, M + c * size, size);
#endif
      for (a = 0; a < N; a++) {
#ifndef BITWISE_DISTANCES
	if (dist > bestd[a]) {
#else
	if (dist < bestd[a]) {
#endif
	  for (d = N - 1; d > a; d--) {
	    bestd[d] = bestd[d - 1];
	    strcpy(bestw[d], bestw[d - 1]);
	  }
	  bestd[a] = dist;
	  strcpy(bestw[a], &vocab[c * max_w]);
	  break;
	}
      }
    }
    if (!strcmp(st4, bestw[0])) {
      CCN++;
      CACN++;
      if (QID <= 5) SEAC++; else SYAC++;
    }
    if (QID <= 5) SECN++; else SYCN++;
    TCN++;
    TACN++;
  }
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(float)TQ*100);
  time = clock() - time;
  printf("Computed accuracy in %f seconds\n", ((double)time) / CLOCKS_PER_SEC);
  return 0;
}
