from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ('Saying Budget violates poll code, Opp to meet EC today', 'neg'),
    ('Sahara gets immunity, tax panel accepts its claim that seized papers not evidence', 'pos'),
    ('EC wants poll candidates to reveal their source of income', 'pos'),
    ('Seeking VRS, chargesheeted Rajendra Kumar says he was told to frame Arvind Kejriwal.', 'neg'),
    ('OK, that is it, says MS Dhoni, quits as ODI, T20 captain', 'neg'),
    ('Amethi man kills 2 women, 8 girls in his family, then hangs himself', 'neg'),
    ('Do not advance Budget, will help BJP: Congress, SP, CPM write to President', 'neg'),
    ('In new CJI JS Khehars bio, unwritten fact: Blood donor every 3 months, for over 40 years', 'pos'),
    ('Justice Khehar takes charge today: Some of  T S Thakurs comments avoidable, says Ravi Shankar Prasad', 'pos'),
    ('Will keep praying for judiciary to be fearless, says T S Thakur', 'pos')
]

cl = NaiveBayesClassifier(train)

blob = TextBlob("Bengaluru molestation case: four arrested.  SC dismisses plea for CBI probe. The biggest ever fire sale of Indian corporate assets has begun, to tide over bad loans crisis", classifier=cl)
print(blob)
print(blob.classify())

for sentence in blob.sentences:
    print(sentence)
    print(sentence.classify())


