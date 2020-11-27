import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def recup_donnees(nb_authors, nb_reviews, data):
    """
    Renvoi le dataframe avec les ids des auteurs et le texte de leur revues
    (pour @nb_authors auteurs et @nb_reviews revues par auteur)
    et renvoi aussi le array avec les valeurs de vérités (ids des auteurs)
    """
    # group by auteurs puis comptage de leur reviews
    tmp = data.groupby(['reviewerID']).size().reset_index(name='counts')
    
    ## ATTENTION : il faut que il y ait assez d'auteurs ayant écrit nb_reviews reviews
    authors = tmp[tmp['counts']>=nb_reviews]['reviewerID'].head(nb_authors)   #les auteurs qu'on va classifier
    
    #on crée le dataframe avec nb_reviews pour chq auteurs de authors, et on ne garde que le reviewerID et le texte
    df = pd.concat([data[data['reviewerID']=='{}'.format(aut)][['reviewerID','reviewText']].head(nb_reviews) for aut in authors])
    
    #on remplace les reviewID par des numéros
    authors_to_label = dict(zip(authors,range(nb_authors)))
    df['reviewerID'].replace(authors_to_label,inplace=True)
    
    #valeurs de vérité
    y_data = np.array(df['reviewerID'])
    return df, y_data



def preparation(df, stem=False, punctuation=False, special_char=False,
                ngram=False, n=3, stop_words=None):
    
    """
    Plusieurs choix de pré-traitement des données : 
        _ stemmatisation
        _ suppression de la ponctuation
        _ suppression des caractères spéciaux
        _ n-gram
        _ suppression des stop words : 'english' si on veut enlever les 
          stopwords et qu'on a du texte en anglais, None sinon
    """
    
    df['reviewText'] = df['reviewText'].map(lambda x: x.lower()) # minuscule
    
    if stop_words != None:
        sw = set(stopwords.words(stop_words)) 
        df['reviewText'] = df['reviewText'].apply(nltk.word_tokenize)
        df['reviewText'] = df['reviewText'].apply(lambda x: [w for w in x if not w in sw])
        df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join(x))
    
    if punctuation:
        df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
    
    if special_char:
        df['reviewText'] = df['reviewText'].str.replace("\\W", ' ')
    
    if stem:
        df['reviewText'] = df['reviewText'].apply(nltk.word_tokenize)
        stemmer = PorterStemmer()
        df['reviewText'] = df['reviewText'].apply(lambda x: [stemmer.stem(y) for y in x])
        df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join(x))
    
    if ngram:
        count_vect = TfidfVectorizer(ngram_range=(n, n),analyzer='char')
    else:
        count_vect = TfidfVectorizer()
    
    return count_vect, df



def Naive_Bayes_Classifier(count_vect, df, y_data, nb_test, taux_train, nb_authors, nb_reviews):
    
    """
    Classification des auteurs à partir des reviews qu'ils ont écrit en 
    utilisant le Naive Bayes 
    """
    
    # count_vect : Vectorizer
    # df : DataFrame
    # y_data : Array contenant les valeurs de vérités (les ids des auteurs)
    # nb_test : nombre de tests à faire
    # taux_train : taux de données en train lors de la séparation train/test
    # nb_authors : nombre d'auteurs à considérer
    # nb_reviews : nombre de revues par auteur à considérer
    
    # output : précision, rappel
    
    counts = count_vect.fit_transform(df['reviewText'])

    ### Validation croisée
    precision = 0
    rappel = 0
    for i in range(nb_test):
        
        # Séparation train/test
        s = int(taux_train*nb_reviews)
        ind_train = []
        ind_test = []
        for i in range(nb_authors):
            ind_test += list(np.random.choice(range(nb_reviews*i,nb_reviews*(i+1)),nb_reviews-s,replace=False))
        ind_train = list(set(range(nb_authors*nb_reviews))-set(ind_test))
    
        X_train = counts[ind_train,:]
        y_train = y_data[ind_train]
        X_test = counts[ind_test,:]
        y_test = y_data[ind_test]
    
        # Apprentissage du modèle
        model = MultinomialNB().fit(X_train, y_train)
    
        # Prediction
        predicted = model.predict(X_test)
        
        # Calcul précision/rappel
        tmp_p = 0
        tmp_r = 0

        for i in range(nb_authors):
            VP = len(np.where((predicted==i) & (y_test==i))[0])
            FP = len(np.where((predicted==i) & (y_test != i))[0])
            FN = len(np.where((predicted != i) & (y_test==i))[0])
            if(VP+FP) != 0:
                tmp_p += (VP/(VP+FP))
            if(VP+FN) != 0:
                tmp_r += (VP/(VP+FN))
            
        tmp_p /= nb_authors
        tmp_r /= nb_authors
        
        precision += tmp_p
        rappel += tmp_r
   
    precision /= nb_test
    rappel /= nb_test  
    
    return precision, rappel


def SVM_Classifier(count_vect, df, y_data, nb_test, taux_train, nb_authors, nb_reviews):
    
    """
    Classification des auteurs à partir des reviews qu'ils ont écrit en 
    utilisant le SVM 
    """
    
    # count_vect : Vectorizer
    # df : DataFrame
    # y_data : Array contenant les valeurs de vérités (les ids des auteurs)
    # nb_test : nombre de tests à faire
    # taux_train : taux de données en train lors de la séparation train/test
    # nb_authors : nombre d'auteurs à considérer
    # nb_reviews : nombre de revues par auteur à considérer
    
    # output : précision, rappel
    
    counts = count_vect.fit_transform(df['reviewText'])

    # Validation croisée
    precision = 0
    rappel = 0
    for i in range(nb_test):
        
        # Séparation train/test
        s = int(taux_train*nb_reviews)
        ind_train = []
        ind_test = []
        for i in range(nb_authors):
            ind_test += list(np.random.choice(range(nb_reviews*i,nb_reviews*(i+1)),nb_reviews-s,replace=False))
        ind_train = list(set(range(nb_authors*nb_reviews))-set(ind_test))
    
        X_train = counts[ind_train,:]
        y_train = y_data[ind_train]
        X_test = counts[ind_test,:]
        y_test = y_data[ind_test]
    
        # Apprentissage du modèle
        model = SVC(kernel='linear', C=1).fit(X_train, y_train)
    
        # Prediction
        predicted = model.predict(X_test)
    
        tmp_p = 0
        tmp_r = 0

        for i in range(nb_authors):
            VP = len(np.where((predicted==i) & (y_test==i))[0])
            FP = len(np.where((predicted==i) & (y_test != i))[0])
            FN = len(np.where((predicted != i) & (y_test==i))[0])
            if(VP+FP) != 0:
                tmp_p += (VP/(VP+FP))
            if(VP+FN) != 0:
                tmp_r += (VP/(VP+FN))
            
        tmp_p /= nb_authors
        tmp_r /= nb_authors
        
        precision += tmp_p
        rappel += tmp_r
   
    precision /= nb_test
    rappel /= nb_test  
    
    return precision, rappel