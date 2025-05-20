
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

# تحميل آمن لموارد NLTK المطلوبة
resources = ['punkt', 'stopwords', 'wordnet']
for res in resources:
    try:
        nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
    except LookupError:
        nltk.download(res)

# تهيئة الأدوات
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# دالة المعالجة المسبقة
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens
