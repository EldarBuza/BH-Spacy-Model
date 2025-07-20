import spacy
from spacy.tokens import Doc, Span, Token
from spacy.training import Example
from spacy.tokens import DocBin
import os

nlp = spacy.blank("hr")

def read_conllu_sentences(file_path):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    return [s for s in content.strip().split("\n\n") if s.strip()]

def parse_conllu_sentence(sentence):
    lines = [line for line in sentence.split("\n") if not line.startswith("#")]
    tokens, upos_tags, xpos_tags, heads, deps, ents = [], [], [], [], [], []
    lemmas = []
    feats_collected = []
    current_ent = None

    for i, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) != 10 or '-' in parts[0] or '.' in parts[0]:
            continue  # Skip malformed or multiword tokens

        id_, form, lemma, upos, xpos, feats, head, deprel, deps_raw, misc = parts
        tokens.append(form)
        upos_tags.append(upos)
        xpos_tags.append(xpos)
        heads.append(int(head) - 1 if head != '0' else i)  # head=0 → root → own index
        deps.append(deprel)
        lemmas.append(lemma)
        feats_collected.append("" if feats == "_" else feats)  # clean missing feats

        ner_tag = "O"
        for field in misc.split("|"):
            if field.startswith("NER="):
                ner_tag = field.split("=")[1]

        if ner_tag.startswith("B-"):
            if current_ent:
                current_ent["end"] = len(tokens) - 1
                ents.append(current_ent)
            current_ent = {"start": len(tokens) - 1, "label": ner_tag[2:]}
        elif ner_tag.startswith("I-") and current_ent:
            continue
        else:
            if current_ent:
                current_ent["end"] = len(tokens)
                ents.append(current_ent)
                current_ent = None

    if current_ent:
        current_ent["end"] = len(tokens)
        ents.append(current_ent)

    return tokens, upos_tags, xpos_tags, heads, deps, ents, lemmas, feats_collected

def build_docs(sentences):
    docs = []
    for sent in sentences:
        tokens, upos, xpos, heads, deps, ents_data, lemmas, feats_list = parse_conllu_sentence(sent)
        doc = Doc(nlp.vocab, words=tokens)

        for i, token in enumerate(doc):
            token.tag_ = xpos[i]
            token.pos_ = upos[i]
            token.dep_ = deps[i]
            token.head = doc[heads[i]]
            token.lemma_ = lemmas[i]
            token.set_morph(feats_list[i])

        spans = []
        for ent in ents_data:
            try:
                span = Span(doc, ent["start"], ent["end"], label=ent["label"])
                spans.append(span)
            except Exception as e:
                print("Skipping invalid span:", ent, e)

        doc.ents = spans
        docs.append(doc)
    return docs

def convert_to_spacy(input_file, output_file):
    sentences = read_conllu_sentences(input_file)
    docs = build_docs(sentences)
    db = DocBin(store_user_data=True)  # important for training!
    for doc in docs:
        db.add(doc)
    db.to_disk(output_file)
    print(f"Saved {len(docs)} docs to {output_file}")

if __name__ == "__main__":
    input_path = "Processed_data/bstrainingdata.conllu"
    output_path = "bs_ready_data.spacy"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    convert_to_spacy(input_path, output_path)

