from flair.embeddings import FlairEmbeddings, BytePairEmbeddings, WordEmbeddings, CharacterEmbeddings

embedding_type = {
    'flair': FlairEmbeddings,
    'bpe': BytePairEmbeddings,
    'we': WordEmbeddings,
    'char': CharacterEmbeddings,
}


def get_embeddings(name):
    splitted = name.split('-', 1)
    if len(splitted) == 2:
        type, path = splitted
        return embedding_type[type](path)
    else:
        type = splitted[0]
        return embedding_type[type]()