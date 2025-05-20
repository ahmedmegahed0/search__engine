import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import os

# أضف /tmp لمسارات NLTK أولاً
nltk_data_path = '/tmp'
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)

# تحميل الموارد بشكل آمن
resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet'
}

for key, path in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(key, download_dir=nltk_data_path)

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
