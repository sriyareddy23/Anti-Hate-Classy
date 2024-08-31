import re, json
from numpy import array 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model
hate_model = load_model('hate_model.h5')
spam_model = load_model('spam_model.h5')

stopwords_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def processed(text) -> list:
    cont_patterns = [
        ('(W|w)on\'t', 'will not'),
        ('(C|c)an\'t', 'can not'),
        ('(I|i)\'m', 'i am'),
        ('(A|a)in\'t', 'is not'),
        ('(\w+)\'ll', '\g<1> will'),
        ('(\w+)n\'t', '\g<1> not'),
        ('(\w+)\'ve', '\g<1> have'),
        ('(\w+)\'s', '\g<1> is'),
        ('(\w+)\'re', '\g<1> are'),
        ('(\w+)\'d', '\g<1> would'),
    ]
    patterns = [(re.compile(regex), repl) for regex, repl in cont_patterns]
    text = text.lower()
    for pattern, repl in patterns: text = re.sub(pattern, repl, text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ' '.join(text.split())
    text = [word for word in word_tokenize(text) if word not in stopwords_set]
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

with open('tokenizer_hate.json') as f: tokenizer_hate = tokenizer_from_json(json.load(f))
with open('tokenizer_spam.json') as f: tokenizer_spam = tokenizer_from_json(json.load(f))

class Model:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
    def predict(self,vec) -> float:
        vec = processed(vec)
        vec = array(self.tokenizer.texts_to_sequences(vec))
        vec = pad_sequences(vec,maxlen=2000)
        return 100*self.model.predict(vec)[0,0]

HateModel = Model(hate_model, tokenizer_hate)
SpamModel = Model(spam_model, tokenizer_spam)


vec = ''
while vec!=['exit']:
    vec = input("[INPUT]: ")
    hate_rating = HateModel.predict(vec)
    spam_rating = SpamModel.predict(vec)
    print(f'{hate_rating:.2f}% hate')
    print(f'{spam_rating:.2f}% spam')





