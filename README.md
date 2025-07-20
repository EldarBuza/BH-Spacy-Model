# Spacy Bosnian language model

This repository contains a custom-trained spaCy pipeline for the Bosnian language, based on high-quality linguistic corpora and carefully preprocessed data. The project aims to fill a gap in Natural Language Processing (NLP) resources for Bosnian by delivering a practical model capable of part-of-speech tagging, lemmatization, dependency parsing, named entity recognition (NER), and sentence segmentation.

## Project description

The project focuses on creating a robust and linguistically rich NLP model for the Bosnian language. It is built using the spaCy framework and leverages datasets derived from relevant South Slavic linguistic resources.

## Problem statement

Although NLP technologies are advancing rapidly, languages with smaller speaker bases—like Bosnian—often lack adequate tools and resources. This project addresses this gap by delivering a functional and extensible Bosnian spaCy model trained on real-world texts.

## Datasets Used

The model is trained on data prepared for inclusion into the Bosnian and Herzegovinian National Corpus (BHNC)
All datasets were transformed to the Universal Dependencies CONLL-U format and then converted into spaCy’s binary .spacy format.

## Technologies & Architecture

* Language: Bosnian (lang = "hr" used in config to leverage Croatian similarity)

* Framework: spaCy

* Pipeline Components:

  * Tagger

  * Parser

  * Lemmatizer

  * Morphologizer

  * Named Entity Recognizer (NER)

  * Sentence Segmenter

* Attribute Ruler

Each component is built using spaCy’s CNN-based HashEmbedCNN encoder for the tok2vec layer.
