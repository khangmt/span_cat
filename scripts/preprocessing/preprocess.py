from spacy.tokens import DocBin, Span
import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm

import pandas as pd

from wasabi import Printer
from wasabi import table

from pathlib import Path
import typer
ABSTAIN = -1
label2id = {
    "OTHER": 0,
    "DATA": 1,
    "DIRECTORY": 2,
    "ENCRYPTION": 3,
    "FUNCTION": 4,
    "NETWORK": 5,
    "COMPONENT": 6,
    "REGISTRY": 7,
    "USER": 8,
    "VULNERABILITY": 9,
    "ACTOR": 10,
    "ABSTAIN": -1
}
id2label = {
    0: "OTHER",
    1: "DATA",
    2: "DIRECTORY",
    3: "ENCRYPTION",
    4: "FUNCTION",
    5: "NETWORK",
    6: "COMPONENT",
    7: "REGISTRY",
    8: "USER",
    9: "VULNERABILITY",
    10: "ACTOR"
}
msg = Printer()
def _character_offset_to_token(doc: Doc, start:int, end:int) -> list:
        token_list = []
        for token in doc:
            if start == token.idx:
                token_list.append(token)
            elif token.idx > start and token.idx < end:
                token_list.append(token)
        return token_list
def _token_offset_to_character(doc: Doc, start, end) -> list:
        char_list = []
        for token in doc:
            if start.i == token.i:
                char_list.append(token.idx)
            if token.i == end.i:
                char_list.append(token.idx + len(token.text))
        return char_list

def main(
    spans_file: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
):
    df = pd.read_csv(spans_file, encoding="utf-8")
    doc_dict = {}
    # nlp = spacy.load('en_core_web_trf')
    nlp = spacy.blank('en')
    # nlp.tokenizer = custom_tokenizer(nlp)
    for index, row in df.iterrows():
        ents = []
        label = id2label[row["label"]]
        phrase = row["phrase"]
        start = row["start"]
        end = row["end"]
        sentence = row["sentence"]
        doc_id = row["hash_id"]
        if sentence[-1]==".":
            sentence = sentence[:-1]
        if sentence not in doc_dict:
            doc_dict[sentence] = nlp(sentence)
        # doc_dict[sentence]["entities"].append((start, end, label))
        token_ids = _character_offset_to_token(doc_dict[sentence], start, end)
        if len(token_ids) == 0:
            print(phrase)
            print("Skipping entity")
            continue
        new_start, new_end = _token_offset_to_character(doc_dict[sentence], token_ids[0], token_ids[-1])
        span = doc_dict[sentence].char_span(new_start, new_end, label=label,alignment_mode='strict')
        if span is None:
            print(sentence)
            print("Skipping entity")
        else:
            ents.append(span)
        doc_dict[sentence].ents = ents
    
    docs = list(doc_dict.values())
    train = []
    dev = []
    split = int(len(docs) * eval_split)
    train = docs[split:]
    dev = docs[:split]

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Processing data done")


if __name__ == "__main__":
    typer.run(main)