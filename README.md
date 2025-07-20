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


## Training Details

* Optimizer: Adam with linear learning rate warm-up
* Learning Rate: 0.00005
* Warmup Steps: 250
* Training Steps: 20,000
* Batch Size: 128
* Dropout: 0.1

Training was performed using GPU acceleration to ensure faster convergence.

## Evaluation Metrics

<img width="495" height="255" alt="metrics" src="https://github.com/user-attachments/assets/9dcf077d-af79-4c97-b49f-70cb57d55a06" />

## Key Achievements

* Successful adaptation of South Slavic resources for Bosnian NLP
* Balanced pipeline with good generalization across syntactic and semantic tasks
* Competitive performance despite limited language resources

## Risks & Considerations

* Lack of large-scale manually annotated Bosnian corpora could affect generalization
* NER performance is lower due to limited annotated named entities in the source data
* Ambiguities specific to Bosnian morphology remain a challenge

## How to Use

1. Clone the repository
2. Install spaCy
3. Download the model (to be published in dist/ or via spacy package)
4. Load the model:

```
import spacy
nlp = spacy.load("bs_Model/model-best")
doc = nlp("Ovo je testna rečenica.")
for token in doc:
    print(token.text, token.pos_, token.lemma_)
```


