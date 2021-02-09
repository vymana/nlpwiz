
import nltk

def _frequent_words(texts, size=500):
    freq_dest = nltk.FreqDist([w.lower() for d in texts for w in str(d).split(" ")])
    common_words = freq_dest.most_common(size)
    return [t[0] for t in common_words if len(t[0])>2 and t[0].isalpha()]


def _document_features(doc, corpus):
    document_words = set(doc.split(' '))
    features = {}
    for word in corpus:
        features['contains({})'.format(word)] = (word in document_words)
    return features


def train(texts, labels):
    corpus = _frequent_words([t for t in texts], size=500)
    doc_features = [_document_features(t, corpus) for t in texts]
    train_data = zip(doc_features, labels)
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    return classifier, corpus


def test(classifier, corpus, texts, labels):
    doc_features = [_document_features(t, corpus) for t in texts]
    predictions = [classifier.classify(d) for d in doc_features]
    num_correct = sum([p==l for p,l in zip(predictions, labels)])
    accuracy = num_correct/len(labels)
    return accuracy


def classify(classifier, corpus, texts):
    doc_features = [_document_features(t, corpus) for t in texts]
    predictions = [classifier.classify(d) for d in doc_features]
    return predictions



if __name__ == "__main__":
    import os
    data_path = os.path.join("test_data", "classification", "imdb_labelled.txt")
    data_lines = [line.strip() for line in open(data_path).readlines()]
    data = [(line[:-1].strip(), int(line[-1])) for line in data_lines]
    l = len(data)
    train_test_split = [0.8, 0.2]
    train_data = data[int(l*train_test_split[0]):]
    test_data = data[:int(l*train_test_split[1])]

    train_texts = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    test_texts = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]

    classifier,corpus = train(train_texts, train_labels)
    accuracy = test(classifier, corpus, test_texts, test_labels)
    print(f"Accuracy: {accuracy}")


