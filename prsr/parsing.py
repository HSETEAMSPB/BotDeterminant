# from config import class_names
import nltk
import spacy
import nest_asyncio
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
nest_asyncio.apply()
def lemmatize(word): # Лемматизация слова
    return lemmatizer.lemmatize(word.lower())


def tokenize_and_label(sentence, class_names):
    sentence1 = sentence.lower()
    tokens = nltk.word_tokenize(sentence1)  # Разделение предложения на токены

    labels = ["O"] * len(tokens)  # Инициализация меток классов предметов как "O" (отсутствие класса)

    for i, token in enumerate(tokens):
        for p in class_names:
            if lemmatize(token) in class_names[p]:
                labels[i] = p  # Используем лемматизированное слово в качестве метки класса

    return tokens, labels


def returner(labels):
    a = []
    for i in labels:
        if i != 'O':
            a.append(i)
    return a


def SearchDoubleWords(doc):
    word = []
    strw = ""
    for i in range(len(doc)):
        if i >= len(doc):
            break
        if doc[i].dep_ == ("compound" or "amod"):
            strw += doc[i].text
            k = i
            while doc[k + 1].dep_ == ("compound" or "amod"):
                strw += " " + doc[k + 1].text
                k += 1
            if doc[k + 1].dep_ != ("compound" or "amod"):
                word.append(strw + " " + doc[k + 1].text)
            strw = ""
            i = k + 1
    return word


def prs(sentence, class_names):
    tokenize_and_label(sentence, class_names)
    tokens, labels = tokenize_and_label(sentence, class_names)
    main_words = returner(labels)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    wordD = SearchDoubleWords(doc)
    need_word = []
    for token in wordD:
        for p in class_names:
            if lemmatize(token) in class_names[p]:
                need_word.append(p)
    for i in main_words:
        need_word.append(i)
    need_word = set(need_word)
    return list(need_word)
