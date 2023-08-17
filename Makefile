BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n ${BASENAME} python=3.10.12 -y
