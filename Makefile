fmt:
	black *.py
	black */*.py

test:
	./tests/gen_test_clips.py
	python3.7 -m unittest


docker_build:
	docker build -qt "chord-detection" .

.PHONY: fmt
