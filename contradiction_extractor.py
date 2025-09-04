import requests
from hazm import POSTagger, Normalizer, WordTokenizer, Lemmatizer, stopwords_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    def __init__(self, text1, text2):
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        self.lemmatizer = Lemmatizer()
        self.text1 = self.normalizer.normalize(text1)
        self.text2 = self.normalizer.normalize(text2)
        self.vectorizer = CountVectorizer()
        self.sentiment_url = "http://37.156.144.109:7083/hugging_face/sentiment_analysis/"
        self.ner_url = "http://37.156.144.109:7083/hugging_face/named_entity_recognition/"
        self.farsnet_url = "http://37.156.144.109:7104/salam/farsnet"
        self.pos_tagger = POSTagger("./models/pos_tagger.model", universal_tag=True)
        self.text1_tags = self.pos_tagger.tag(self.tokenizer.tokenize(self.text1))
        self.text2_tags = self.pos_tagger.tag(self.tokenizer.tokenize(self.text2))
        self.stopwords = set(stopwords_list())

    def get_sentiment(self, text):
        data = {
            "input_text": text
        }
        try:
            response = requests.post(self.sentiment_url, json=data)
            sentiment = response.json()[0]
        except:
            sentiment = "neutral"
        if sentiment == "very positive":
            return 1
        elif sentiment == "positive":
            return 0.7
        elif sentiment == "negative":
            return 0.3
        elif sentiment == "very negative":
            return 0
        return 0.5

    def get_name_entities(self, text):
        data = {
            "input_text": text
        }
        try:
            response = requests.post(self.ner_url, json=data)
            response = response.json()
        except:
            return [0 for _ in range(10)]
        result = {}
        for entity_type, entities_info in response.items():
            result[entity_type] = []
            for entity_info in entities_info:
                result[entity_type].append(entity_info["word"])
        return result

    def compare_name_entities(self):
        text1_entities = self.get_name_entities(self.text1)
        text2_entities = self.get_name_entities(self.text2)
        result = []
        for entity_type in text1_entities.keys():
            if len(text1_entities[entity_type]) >=1 and len(text2_entities[entity_type]) >= 1:
                text1_entity = set(text1_entities[entity_type])
                text2_entity = set(text2_entities[entity_type])
                union = text1_entity.union(text2_entity)
                intersect = text1_entity.intersection(text2_entity)
                result.append(len(union - intersect))
            else:
                result.append(0)
        return result

    def pos_compare(self, pos):
        text1_adjs = set([self.lemmatizer.lemmatize(word_pos[0]) for word_pos
                          in self.text1_tags if word_pos[1] == pos])
        text2_adjs = set([self.lemmatizer.lemmatize(word_pos[0]) for word_pos
                          in self.text2_tags if word_pos[1] == pos])
        union = text1_adjs.union(text2_adjs)
        intersect = text1_adjs.intersection(text2_adjs)
        return len(union - intersect)

    def negation_check(self):
        text1_verbs = set([self.lemmatizer.lemmatize(word_pos[0]) for word_pos
                          in self.text1_tags if word_pos[1] == "VERB"])
        text2_verbs = set([self.lemmatizer.lemmatize(word_pos[0]) for word_pos
                          in self.text2_tags if word_pos[1] == "VERB"])
        union = text1_verbs.union(text2_verbs)
        intersect = text1_verbs.intersection(text2_verbs)
        delta = union - intersect
        for i in range(len(delta)):
            for j in range(i+1, len(delta)):
                if (delta[i].startswith("ن") and delta[i][1:] == delta[j]) \
                        or (delta[j].startswith("ن") and delta[j][1:] == delta[i]):
                    return 1
        return 0

    def common_words(self):
        text1_words = set(self.tokenizer.tokenize(self.text1))
        text2_words = set(self.tokenizer.tokenize(self.text2))
        return len(text1_words.union(text2_words))

    def cosine_similarity(self):
        text1_edited = set(self.text1.split(" ")) - self.stopwords
        text2_edited = set(self.text2.split(" ")) - self.stopwords
        text1_edited = [self.lemmatizer.lemmatize(word) for word in text1_edited]
        text2_edited = [self.lemmatizer.lemmatize(word) for word in text2_edited]
        text1 = ""
        for word in text1_edited:
            text1 += word + " "
        text1 = text1.strip()
        text2 = ""
        for word in text2_edited:
            text2 += word + " "
        text2 = text2.strip()
        corpus = [text1, text2]
        vectors = self.vectorizer.fit_transform(corpus)
        similarity_matrix = cosine_similarity(vectors)
        return similarity_matrix[0][1]

    def check_antonym(self):
        text1_edited = set(self.tokenizer.tokenize(self.text1)) - self.stopwords
        text2_edited = set(self.tokenizer.tokenize(self.text2)) - self.stopwords
        text1_edited = [self.lemmatizer.lemmatize(word) for word in text1_edited]
        text2_edited = [self.lemmatizer.lemmatize(word) for word in text2_edited]
        for word1 in text1_edited:
            for word2 in text2_edited:
                try:
                    request_url = f"{self.farsnet_url}/?word1={word1}&word2={word2}&type=Antonym"
                    response = requests.get(request_url)
                    response = response.json()
                except:
                    response = {"total_hits": 0}
                if response["total_hits"] > 0:
                    return 1
        return 0

    def feature_construction(self):
        text1_sentiment = self.get_sentiment(self.text1)
        text2_sentiment = self.get_sentiment(self.text2)
        name_entity_features = self.compare_name_entities()
        text1_length = len(self.text1.split(" "))
        text2_length = len(self.text2.split(" "))
        adj_compare = self.pos_compare("ADJ")
        verb_compare = self.pos_compare("VERB")
        num_compare = self.pos_compare("NUM")
        no_common_words = self.common_words()
        similarity = self.cosine_similarity()
        #antonym = self.check_antonym()
        feature_vector = [text1_sentiment, text2_sentiment, text1_length, text2_length,
                adj_compare, verb_compare, num_compare, no_common_words, similarity]
        for val in name_entity_features:
            feature_vector.append(val)
        return feature_vector
