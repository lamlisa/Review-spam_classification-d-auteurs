import numpy as np

############################# REVIEW GRAPH ####################################


# reviewID : un numéro représentant l'identifiant d'une revue, identifié par 
# son numéro de ligne dans la base de données
#
# reviewerID : un numéro représentant l'identifiant d'un reviewer
#
# productID : un numéro représentant l'identifiant d'un produit
#
# reviewer_reviews : {reviewerID : [reviewsID]}
# les identifiants des reviewers et la liste des identifiants de ses revues
#
# product_reviews : {productID : [reviewsID]}
# les identifiants des produits et la liste des identifiants des revues qu'il a écrit
#
# review_author = array de l'auteur de la revue
#
# review_product = array du produit de la revue
#
# time_post = array du temps à laquelle la revue a été postée (en secondes)
#
# notes = array des notes de chaque revue
#
# avg_notes : {productID : avg_note}
#
# honesty : array contenant les valeurs d'honêteté de chaque revue
#
# trustiness : array contenant les valeurs de confiance de chaque auteur
#
# reliability : array contenant les valeurs de fiabilité de chaque produit
#
# reviewsID = array contenant les numéros représentant les ID des revues
#
# reviewersID = array contenant les numéros représentant les ID des auteurs
#
# productsID = array contenant les numéros représentant les ID des produits


def reviewer_trustiness(reviewerID, reviewer_reviews, honesty):
    """
    int*dict*array -> float
    output : the trustiness of the reviewer
    """
    #somme des honesty score de toutes les revues du reviewer
    sum_honesty_reviews = np.sum(honesty[reviewer_reviews[reviewerID]])
    return 2/(1+np.exp(-sum_honesty_reviews))-1


def surrounding_set(reviewID, product_reviews, delta_t, time_post,
                    review_product):
    """
    output : l'ensemble contenant les reviews postées entre delta_t secondes avant ou 
             après que la review ait été postée    
    """
    # temps où la review a été postée
    tv = time_post[reviewID]
    
    # reviews du même produit
    reviews = product_reviews[review_product[reviewID]]
    
    # indices des revues de reviews qu'on veut
    tmp = np.where(np.abs(time_post[reviews]-tv) <= delta_t)[0]
    
    return reviews[tmp]
 
    
def agree_disagree_set(reviewID, product_reviews, delta_t, delta, notes,
                       time_post, review_product):
    """
    output : 2 ensembles -> les identifiants des reviews qui sont dans la 
             fenêtre de temps et qui soit sont d'accord ou pas avec la review
    """
    Sv = surrounding_set(reviewID, product_reviews, delta_t, time_post,
                         review_product)
    notes_voisins = notes[Sv]
    tmp = np.where(abs(notes_voisins-notes[reviewID]) <= delta)[0]
    return Sv[tmp], list(set(Sv)-set(Sv[tmp]))
  
    
def review_agreement(reviewID, product_reviews, trustiness, delta_t, delta,
                     review_author, notes, time_post, review_product):
    """
    output : le niveau d'accord entre les revues voisines (en terme de temps) 
             du même produit
    """
    Sva, Svd = agree_disagree_set(reviewID, product_reviews, delta_t, delta,
                                  notes,time_post,review_product)
    
    # somme des scores de trustiness des reviewers qui étaient en accord avec la 
    # review, et l'on soustrait la somme des scores de ceux en désaccord
    agreement = sum([trustiness[review_author[i]] for i in Sva]) - sum([trustiness[review_author[i]] for i in Svd])
    
    # normalisation entre -1 et 1
    return 2/(1+np.exp(-agreement))-1


def review_honesty(reviewID, agreement, reliability, review_product):
    """
    output : le score d'honnêteté d'une revue
    """
    return abs(reliability[review_product[reviewID]])*agreement[reviewID]


def product_reliability(productID, avg_notes, product_reviews, reviewer_reviews,
                        trustiness, review_author, notes):
    """
    output : le score de fiabilité d'un produit
    """
    # les revues de ce produit
    p_reviews = product_reviews[productID]
    
    # trustiness des auteurs des revues de ce produit
    trust = trustiness[review_author[p_reviews]]
    
    # indices des auteurs de ce produit dont le trust est positif
    pos_trust = np.where(trust > 0)[0]
    
    teta = sum(trust[pos_trust]*(notes[p_reviews[pos_trust]]-avg_notes[productID]))
    return 2/(1+np.exp(-teta))-1  


def compute(productsID,reviewsID,reviewersID,nb_rounds,product_reviews,reviewer_reviews,delta_t,delta,avg_notes,review_author,notes,time_post,review_product):
    """
    output : les 3 arrays contenant respectivement les scores de fiabilité d'un produit,
             les scores de confiance d'un auteur et les scores d'honnêteté d'une revue
    """
    honesty = np.zeros(len(reviewsID))
    reliability = np.ones(len(productsID))
    trustiness = np.ones(len(reviewersID))
    agreement = np.zeros(len(reviewsID))
    
    for re in reviewsID:
        agreement[re] = review_agreement(re,product_reviews,trustiness,delta_t,delta,review_author,notes,time_post,review_product)
    
    count = 0
    while count < nb_rounds:
        
        for re in reviewsID:
            honesty[re] = review_honesty(re,agreement,reliability,review_product)
        
        for r in reviewersID:
            trustiness[r] = reviewer_trustiness(r,reviewer_reviews,honesty)

        for p in productsID:
            reliability[p] = product_reliability(p,avg_notes,product_reviews,reviewer_reviews,trustiness,review_author,notes)

        for re in reviewsID:
            agreement[re] = review_agreement(re,product_reviews,trustiness,delta_t,delta,review_author,notes,time_post,review_product)
        count += 1
        
    return reliability, trustiness, honesty
