from spacy.tokens import DocBin, Span
import spacy
from spacy.tokens.doc import Doc
from tqdm import tqdm

import pandas as pd

from wasabi import Printer
from wasabi import table

from pathlib import Path
import typer
ACTOR = 0
COMMUNICATION = 1
DATA = 2
DIRECTORY = 3
DLL = 4
ENCRYPTION = 5
FUNCTION = 6
NETWORK = 7
COMPONENT = 8
REGISTRY = 9
USER = 10
VULNERABILITY = 11
OTHER = 12
id2label = {
    0: "ACTOR",
    1: "COMMUNICATION",
    2: "DATA",
    3: "DIRECTORY",
    4: "DLL",
    5: "ENCRYPTION",
    6: "FUNCTION",
    7: "NETWORK",
    8: "COMPONENT",
    9: "REGISTRY",
    10: "USER",
    11: "VULNERABILITY",
    12: "OTHER",
}
msg = Printer()
def _character_offset_to_token(doc: Doc, start:int, end:int) -> list:
        token_list = []
        for token in doc:
            if start == token.idx:
                token_list.append(token.i)
            elif token.idx > start and token.idx <= end:
                token_list.append(token.i)
        return token_list

def main(
    spans_file: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    span_key: str,
):
    nlp = spacy.blank("en")
    msg.info("Processing data")
    df = pd.read_csv(spans_file)
    doc_dict = {}
    for index, row in df.iterrows():
        label = id2label[row["label"]]
        span = row["phrase"]
        start = row["start"]
        end = row["end"]
        sentence = row["sentence"]
        doc_id = row["hash_id"]
        if doc_id not in doc_dict:
            doc_dict[doc_id] = nlp(sentence)
            doc_dict[doc_id].spans[span_key] = []
        
        doc = doc_dict[doc_id]
        tokens = _character_offset_to_token(doc, start, end)
        if not tokens:
            continue
        span_object = Span(doc, tokens[0], tokens[-1]+1, label=label)
        doc.spans[span_key].append(span_object)
    
    docs = list(doc_dict.values())
    train = []
    dev = []
    split = int(len(docs) * eval_split)
    train = docs[split:]
    dev = docs[:split]

    # Get max span length from train
    max_span_length = 0
    for doc in train:
        for span in doc.spans[span_key]:
            span_length = span.end - span.start
            if span_length > max_span_length:
                max_span_length = span_length

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Processing data done")


if __name__ == "__main__":
    typer.run(main)