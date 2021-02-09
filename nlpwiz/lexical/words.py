import textstat
from wordfreq import word_frequency,zipf_frequency
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = None

def get_word_stats(word):
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentIntensityAnalyzer()

    count_syllables = textstat.syllable_count(word)
    freq_score = zipf_frequency(word, "en")
    polarity = sentiment_analyzer.polarity_scores(word)
    stats = {
        "syllables": count_syllables,
        "freq_score": freq_score,
        "sentiment": 1 if polarity["pos"] else -1 if polarity["neg"] else 0,
        "sentiment_degree": polarity["compound"],
        "difficulty": (min(count_syllables, 6) * 5 // (1 + min(freq_score, 6)))
    }
    return stats
