import gzip
import pandas as pd
import numpy as np


# Créer les variables nécessaires pour le review graph à partir des données
# Amazon Instant Video : https://jmcauley.ucsd.edu/data/amazon/


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Amazon_Instant_Video.json.gz')

df2 = df.head(100000)

#remplacer les reviewerID par des numéros
tmp = sorted(set(df2['reviewerID']))
reviewerID_to_label = dict(zip(tmp,range(len(tmp))))
df2['reviewerID'] = df2['reviewerID'].map(reviewerID_to_label)

#remplacer les asin par des numéros
tmp = sorted(set(df2['asin']))
asin_to_label = dict(zip(tmp,range(len(tmp))))
df2['asin'] = df2['asin'].map(asin_to_label)

product_reviews = df2.groupby(['asin']).groups

reviewer_reviews = df2.groupby(['reviewerID']).groups

avg_notes = df2.groupby(['asin']).mean()['overall'].to_dict()

review_author = np.array(df2['reviewerID'])

review_product = np.array(df2['asin'])

time_post = np.array(df2['unixReviewTime'])

notes = np.array(df2['overall'])

reviewsID = np.arange(0,len(df2))

reviewersID = np.arange(0,len(reviewer_reviews))

productsID = np.arange(0,len(product_reviews))
    