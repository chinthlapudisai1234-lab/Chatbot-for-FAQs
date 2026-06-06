#nltk.download('punkt_tab')
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

faq_questions = [
    "What is your return policy?",
    "How can I track my order?",
    "Do you offer international shipping?"
]

faq_answers = [
    "Returns are accepted within 30 days.",
    "Use the tracking link sent to your email.",
    "Yes, we ship worldwide."
]

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    [preprocess(q) for q in faq_questions]
)

def get_best_match(user_query):
    query_vector = vectorizer.transform(
        [preprocess(user_query)]
    )

    similarity = cosine_similarity(
        query_vector,
        faq_vectors
    )

    max_score = similarity.max()

    if max_score < 0.2:
        return None

    return similarity.argmax()

def chatbot(user_query):
    index = get_best_match(user_query)

    if index is None:
        return "Sorry, I couldn't find an answer to that question."

    return faq_answers[index]

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Bot: Goodbye!")
        break

    print("Bot:", chatbot(question))