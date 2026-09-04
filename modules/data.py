from datasets import load_dataset
import re

from torch.utils.data import DataLoader
import torch

from modules import ceaser


BLOCK_LENGTH = 32
BATCH_SIZE = 256


def clean_text(sentence):
    text = sentence["text"].lower()
    text = re.sub(r"[^a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentence["text"] = text
    return sentence


def truncate(sentence):
    # Truncate the text to BLOCK_LENGTH characters
    sentence["text"] = sentence["text"][:BLOCK_LENGTH]
    return sentence


def char_to_num(char):
    # Convert a character to a number
    assert len(
        char) == 1, "Error, length of string passed to char_to_num() should be 1"
    if char.isalpha():
        return ord(char) - ord("a") + 1
    elif char == " ":
        return 0
    else:
        assert False, "Error, char_to_num() only accepts alphabet or space characters"


def num_to_char(num):
    # Convert a number to a character
    assert isinstance(num, int), "Error, num_to_char() only accepts integers"
    if num == 0:
        return " "
    elif 1 <= num <= 26:
        return chr(num + ord("a") - 1)
    else:
        assert False, "Error, num_to_char() only accepts integers between 0 and 26"


def tokenise(sentence):
    sentence["tokens"] = [char_to_num(char) for char in sentence["text"]]
    return sentence


def untokenise(tokens):
    return "".join([num_to_char(token) for token in tokens])


def get_data(limit=2**15):
    data = load_dataset("agentlans/high-quality-english-sentences")

    data["train"] = data["train"].select(range(limit))
    data["test"] = data["test"].select(range(limit))

    data = data.map(clean_text)
    data = data.filter(lambda sentence: len(sentence["text"]) >= BLOCK_LENGTH)
    data = data.map(truncate)
    data = data.map(tokenise)
    data = data.map(
        lambda sentence: {
            "encrypted_tokens": ceaser.encrypt(sentence["tokens"])
        }
    )

    return data["train"], data["test"]


def create_dataloader(data, batch_size=BATCH_SIZE, shuffle=True):
    data.set_format(type="torch", columns=["encrypted_tokens", "tokens"])
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
