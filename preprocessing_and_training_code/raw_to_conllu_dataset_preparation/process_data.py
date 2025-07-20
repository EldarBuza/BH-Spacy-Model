import spacy
import os
import re

# Load Croatian model
try:
    nlp = spacy.load("hr_core_news_sm")
    print("Croatian model loaded.")
except:
    nlp = spacy.blank("hr")
    print("Croatian model not found. Using blank fallback model.")

def fix_deprel(token):
    if token.dep_ == "nummod" and token.head.pos_ == "NOUN":
        return "nummod:gov"
    if token.dep_ == "ROOT":
        return "root"
    return token.dep_ if token.dep_ else "dep"

def find_head(token):
    return 0 if token.dep_ == "ROOT" else token.head.i + 1

def sanitize(value):
    return str(value).replace("\t", " ").replace("\n", " ").strip() or "_"

def to_conllu(doc, sent_id):
    lines = [f"# sent_id = {sent_id}", f"# text = {doc.text}"]
    for i, token in enumerate(doc, start=1):
        form = sanitize(token.text)
        lemma = sanitize(token.lemma_)
        upos = token.pos_ or "X"
        xpos = token.tag_ or "_"
        feats = sanitize(token.morph)
        head = find_head(token)
        deprel = fix_deprel(token)

        if token.ent_type_:
            if token.ent_iob_ == "B":
                ner = f"B-{token.ent_type_}"
            elif token.ent_iob_ == "I":
                ner = f"I-{token.ent_type_}"
            else:
                ner = "O"
        else:
            ner = "O"

        misc = [f"NER={ner}"]
        if token.ent_type_:
            misc.append("NamedEntity=Yes")
        if i < len(doc) and not token.whitespace_:
            misc.append("SpaceAfter=No")

        misc_str = "|".join(misc) if misc else "_"

        lines.append(f"{i}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\t{misc_str}")

    lines.append("")
    return "\n".join(lines)

def extract_articles(text):
    articles = re.split(r"<\*{3,}>", text)
    cleaned_articles = []
    for art in articles:
        match = re.search(r"AUTOR\(I\):.*?\n\n(.+)", art.strip(), flags=re.DOTALL)
        if match:
            body = match.group(1).strip()
            cleaned_articles.append(body)
    return cleaned_articles

def process_and_append(input_path, output_path, file_index):
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    articles = extract_articles(raw)
    conllu_blocks = []

    # Batch process for speed
    docs = list(nlp.pipe(articles, batch_size=16, n_process=4))

    for idx, doc in enumerate(docs):
        for i, sent in enumerate(doc.sents):
            sentence_doc = nlp(sent.text)
            conllu_blocks.append(to_conllu(sentence_doc, sent_id=f"file{file_index}_art{idx+1}_sent{i+1}"))

    with open(output_path, "a", encoding="utf-8") as out_f:
        out_f.write("\n".join(conllu_blocks))
        out_f.write("\n")

if __name__ == "__main__":
    input_folder = "Raw_data/"
    output_file_path = "Processed_data/bstrainingdata.txt"

    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            full_input_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            process_and_append(full_input_path, output_file_path, idx + 1)

    print("All files processed and appended to:", output_file_path)
