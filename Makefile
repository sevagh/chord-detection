fmt:
	black *.py
	black */*.py

docker_build:
	docker build -qt "chord-detection" .

.PHONY: fmt
