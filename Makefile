
all:
	make -C src

clean:
	make -C src clean

rebuild:
	rm -fr ./data/positive.samples/words ./data/unlabeled.samples/words ./data/test.samples/words && \
	./bin/udb --rebuild --samples_dir ./data/positive.samples && \
	./bin/udb --rebuild --samples_dir ./data/unlabeled.samples && \
	./bin/udb --rebuild --samples_dir ./data/test.samples

learning:
	./bin/udb --learning --positive_samples ./data/positive.samples --unlabeled_samples ./data/unlabeled.samples --test_samples ./data/test.samples


predict:
	./bin/udb --predict --samples_dir ./data/test.samples


