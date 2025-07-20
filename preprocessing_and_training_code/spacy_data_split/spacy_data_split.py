import random
import spacy
from spacy.tokens import DocBin

# Load your single .spacy file
input_path = "bs_ready_data.spacy"
train_output_path = "Spacy_proper_split_data/bs_train_data_v2.spacy"
dev_output_path = "Spacy_proper_split_data/bs_dev_data_v2.spacy"

# Load the DocBin
doc_bin = DocBin().from_disk(input_path)
docs = list(doc_bin.get_docs(spacy.blank("bs").vocab))

# Shuffle and split
random.shuffle(docs)
split = int(len(docs) * 0.7)  # 70% train, 30% dev

train_docs = docs[:split]
dev_docs = docs[split:]

# Save new .spacy files
DocBin(docs=train_docs).to_disk(train_output_path)
DocBin(docs=dev_docs).to_disk(dev_output_path)

print(f"Split complete: {len(train_docs)} train docs, {len(dev_docs)} dev docs")
