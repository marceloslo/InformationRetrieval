import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import string

#class responsible for parsing
class Parser:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stem = SnowballStemmer("english").stem
        self.tokenize = word_tokenize
        self.stopw = set(stopwords.words('english'))
        self.remove_digits = str.maketrans(dict.fromkeys(string.digits, " "))
        self.remove_punctuation = str.maketrans(dict.fromkeys(string.punctuation, " "))
        #regex pattern for keeping (mostly) characters that appear in english, including diacritics
        self.non_english_chars = re.compile(r'[^\u0030-\u007A\u00C0-\u00F6\u00F8-\u00FF]+')
    def parse(self,sentence):
        sentence = sentence.translate(self.remove_digits)
        sentence = sentence.translate(self.remove_punctuation)
        #sentence = self.keep_english_chars.sub(' ',sentence)
        tokens = self.tokenize(sentence)
        #remove words with (most) characters that cannot appear in english language, stem and lower
        tokens = [self.stem(w.lower()) for w in tokens if (w.lower() not in self.stopw) and (self.non_english_chars.search(w) is None)]
        return tokens
    def freq_dist(self,sentence):
        tokens = self.parse(sentence)
        return len(tokens),nltk.FreqDist(tokens).items()