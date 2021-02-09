NLPWiz Package
==========================


## About

NLP Tools on the go


## Requirements

Package requirements are handled using pip. To install them do

```
pip install -r requirements.txt

#download spacy model
python -m spacy download en

#download nltk wordnet
python -c "import nltk; nltk.download('wordnet')"


```

## Usage

```python
from nlpwiz import nlp

TEST_TEXT = "Jack went to New York by the morning flight"

# nlp parsing
tags = nlp.parse(TEST_TEXT)


# lemmatization
lemmas = nlp.lemmatize(TEST_TEXT) 

```


## Tests

Testing is set up using [pytest](http://pytest.org) and coverage is handled
with the pytest-cov plugin.

Run your tests with ```py.test``` in the root directory.

Coverage is ran by default and is set in the ```pytest.ini``` file.
To see an html output of coverage open ```htmlcov/index.html``` after running the tests.

