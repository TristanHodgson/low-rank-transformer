from random import randint

def key_encrypt(tokens, key):
    return [
        ((token - 1 + key) % 26) + 1 if token != 0 else 0
        for token in tokens
    ]


def encrypt(tokens):
    key = randint(1, 25)
    return key_encrypt(tokens, key)