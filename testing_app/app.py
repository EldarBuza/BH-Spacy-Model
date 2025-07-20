import spacy
import gradio as gr
from spacy import displacy

nlp = spacy.load("BS-model/model-best")

def analyze(text):
    doc = nlp(text)
    # Visual NER as HTML
    ner_html = displacy.render(doc, style="ent", jupyter=False)
    
    # POS & Lemma Table
    pos_table = [(token.text, token.pos_, token.lemma_) for token in doc]
    
    return ner_html, pos_table

iface = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, label="Unesi rečenicu"),
    outputs=[
        gr.HTML(label="Prepoznati entiteti (vizualno)"),
        gr.Dataframe(headers=["Token", "POS", "Lemma"])
    ],
    title="Bosanski NLP Analizator",
    description="Unesi rečenicu na bosanskom jeziku kako bi se prikazali entiteti i lingvističke informacije."
)

iface.launch()
