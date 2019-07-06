#!/bin/bash

docker build . -t seldon-core-s2i-python3-spacy:0.6 
s2i build . seldon-core-s2i-python3-spacy:0.6 spacy_tokenizer:0.1
# s2i build . seldonio/seldon-core-s2i-python37:0.10 spacy_tokenizer:0.1

