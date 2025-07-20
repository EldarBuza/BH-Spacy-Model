import spacy
nlp = spacy.load("BS-Model/model-best")

doc = nlp("Ovo je primjer rečenice za testiranje modela.")

print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text:<30} {ent.label_}")