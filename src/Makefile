
all: jieba_test

jieba_test: jieba_test.o
	${CC} -o jieba_test jieba_test.o -lstdc++ -lm

jieba_test.o: jieba_test.cpp
	${CC} -c jieba_test.cpp -I../include/jieba

