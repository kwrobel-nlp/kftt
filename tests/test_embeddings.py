import torch

from flair.data import Token, Sentence
from flair.embeddings import FlairEmbeddings

from tokenization import FlairEmbeddingsEnd, FlairEmbeddingsStart, FlairEmbeddingsOuter


def test_original():
    ff = FlairEmbeddings('pl-forward')
    fb = FlairEmbeddings('pl-backward')
    
    for whitespace_after in [False, True]:
        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s1)
    
        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('Xota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)
    
        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()
    
        assert all(torch.eq(a, b))
    
        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('mX', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)
    
        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()
    
        assert not all(torch.eq(a, b))
    
        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s1)
    
        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('AlX', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)
    
        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()
    
        assert all(torch.eq(a, b))
    
        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('Xa', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)
    
        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()
    
        assert not all(torch.eq(a, b))


def test_end():
    ff = FlairEmbeddingsEnd('pl-forward')
    fb = FlairEmbeddingsEnd('pl-backward')

    for whitespace_after in [False, True]:
        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('Xota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[1].get_embedding()
        b = s2.tokens[1].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('mX', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))

        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('mX', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('Xota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))
        
def test_start():
    ff = FlairEmbeddingsStart('pl-forward')
    fb = FlairEmbeddingsStart('pl-backward')

    for whitespace_after in [False, True]:
        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('Xa', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('AlX', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))

        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('AlX', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('Xa', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))
        
def test_outer():
    ff = FlairEmbeddingsOuter('pl-forward')
    fb = FlairEmbeddingsOuter('pl-backward')

    for whitespace_after in [False, True]:
        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('Xa', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('AlX', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        ff.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))

        s1 = Sentence()
        s1.add_token(Token('1', whitespace_after=whitespace_after))
        s1.add_token(Token('2', whitespace_after=whitespace_after))
        s1.add_token(Token('Ala', whitespace_after=whitespace_after))
        s1.add_token(Token('ma', whitespace_after=whitespace_after))
        s1.add_token(Token('kota', whitespace_after=whitespace_after))
        s1.add_token(Token('3', whitespace_after=whitespace_after))
        s1.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s1)

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('mX', whitespace_after=whitespace_after))
        s2.add_token(Token('kota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert all(torch.eq(a, b))

        s2 = Sentence()
        s2.add_token(Token('1', whitespace_after=whitespace_after))
        s2.add_token(Token('2', whitespace_after=whitespace_after))
        s2.add_token(Token('Ala', whitespace_after=whitespace_after))
        s2.add_token(Token('ma', whitespace_after=whitespace_after))
        s2.add_token(Token('Xota', whitespace_after=whitespace_after))
        s2.add_token(Token('3', whitespace_after=whitespace_after))
        s2.add_token(Token('4', whitespace_after=whitespace_after))
        fb.embed(s2)

        a = s1.tokens[3].get_embedding()
        b = s2.tokens[3].get_embedding()

        assert not all(torch.eq(a, b))