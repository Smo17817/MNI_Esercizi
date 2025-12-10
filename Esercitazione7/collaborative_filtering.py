
from recommendation_data import dataset
from math import sqrt
import numpy as np
import pandas as pd

# calcola il punteggio di similarità tra due utenti utilizzando la distanza euclidea
def euclidean_similarity(person1, person2):
	both_viewed = [item for item in dataset[person1] if item in dataset[person2]]

	if len(both_viewed) == 0:
		return 0
	
	euclidean_distance = sqrt(sum([pow(dataset[person1][item] - dataset [person2][item], 2) for item in both_viewed]))

	# la similarità euclidea è pari a 1 / (1 + distanza euclidea)
	return 1 / (1 + euclidean_distance)

# calcola la Correlazione di Pearson tra due utenti
def pearson_correlation(person1,person2):

	# genera la lista degli elementi valutati da entrambi gli utenti
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

	number_of_ratings = len(both_rated)		
	
	# controlla se non ci sono elementi valutati in comune
	if number_of_ratings == 0:
		return 0

	# calcola la somma delle preferenze di ciascun utente 
	person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

	# calcola la somma dei quadrati delle preferenze di ciascun utente
	person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

	# somma del prodotto delle preferenze di entrambi gli utenti
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

	# calcolo della correlazione di Pearson
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r 

# Restituisce il numero di utenti più simili per un dato utente specifico
def most_similar_users(person,number_of_users):
	# viene calcolata la correlazione di Pearson tra l'utente specifico e tutti gli altri utenti nel dataset (escluso se stesso)
	scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset if  other_person != person ]
	
	# ordina i punteggi di somiglianza in ordine decrescente e restituisce i primi 'number_of_users' utenti più simili
	scores.sort()
	scores.reverse()
	return scores[0:number_of_users]

# Genera raccomandazioni per un utente in base alla media ponderata delle valutazioni di tutti gli altri utenti
def user_reommendations(person, correlation=pearson_correlation):
	totals = {}
	simSums = {}
	rankings=[]

	# confronta l'utente attuale con ogni altro utente nel dataset
	for other in dataset:
		# se l'utente è se stesso, salta
		if other == person:
			continue

		# calcola la somiglianza tra ciascun utente e l'utente attuale
		sim = correlation(person,other)

		# ignora punteggi di somiglianza zero o negativi
		if sim <=0: 
			continue

		# per ogni film valutato da other_user
		for item in dataset[other]:
			# assegna un punteggio solo se l'utente non ha ancora visto il film
			if item not in dataset[person] or dataset[person][item] == 0:
				# Similarity * Score
				totals.setdefault(item,0)
				totals[item] += dataset[other][item]* sim
				# Somma delle similarità
				simSums.setdefault(item,0)
				simSums[item]+= sim

	# crea la lista delle raccomandazioni
	rankings = [(total/simSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()
	# returns the recommended items
	recommendataions_list = [recommend_item for score,recommend_item in rankings]
	return recommendataions_list

def prediction_function1(person, item, total, sim_sum):
	return total / sim_sum

def prediction_function2(person, item, total, sim_sum):
	# media di tutti i rating di person
    mu_k = np.mean(list(dataset[person].values()))

    numerator = 0
    denominator = 0

    # scorri tutti gli utenti j che hanno valutato l'item
    for other in dataset:
        # salta se l'utente è se stesso o se non ha valutato l'item
        if other == person  or item not in dataset[other]:
            continue  

        # similarità tra person e other (Pearson)
        sim = pearson_correlation(person, other)
        # si ignorano similarità negative o nulle
        if sim <= 0:
            continue  

        # media dell'utente j
        mu_j = np.mean(list(dataset[other].values()))

        # aggiorno numeratore e denominatore
        numerator += sim * (dataset[other][item] - mu_j)
        denominator += sim

    # se il denominatore è 0, non si può stimare
    if denominator == 0:
        return 'Error'

    # formula finale
    return mu_k + (numerator / denominator)


def create_recommendation_table(person, correlation=pearson_correlation, prediction_function=prediction_function1):
    # calcola i film raccomandati per l'utente
    recommended_items = user_reommendations(person, correlation)

    # Critici con similarità > 0
    critics = []  
    sims = {}

    for other in dataset:
		# se l'utente è se stesso, salta
        if other == person:
            continue

        sim = correlation(person, other)
		# ignora punteggi di similarità zero o negativi
        if sim <= 0:
            continue  

		# aggiungi critico e similarità
        critics.append(other)
        sims[other] = sim

    ### Costruzione label colonne ###
	# Porzione statica
    columns = ['Critic', 'Similarity']
	# porzione dinamica - dipende dai film raccomandati
    for item in recommended_items:
        columns.append(item) # nome del film
        columns.append(f'S.x {item}') # Similarity * rating

    ### Costruzione righe - critici con similarità > 0 ###
    rows = []

    for critic in critics:
        sim = round(sims[critic], 2) # arrontonda similarità a 2 decimali
        row = {
            'Critic': critic,
            'Similarity': sim
        }

        for item in recommended_items:
            # rating dato dal critico al film (se esiste)
            rating = dataset[critic].get(item, '')
            row[item] = rating

            # Similarity * rating (solo se il rating esiste)
            if rating != '': 
                row[f'S.x {item}'] = round(sim * rating, 2)
            else:
                row[f'S.x {item}'] = ''

        rows.append(row)

    ### Creazione DataFrame ###	
    df = pd.DataFrame(rows, columns=columns)

    ### Calcolo: Total, SimSum, Total/SimSum (aggiunti alla colonna dei critici, con similarità vuota) ###
    total_row = {'Critic': 'Total', 'Similarity': ''}
    sim_sum_row = {'Critic': 'Sim.Sum', 'Similarity': ''}
    pred_row = {'Critic': 'Total/Sim.Sum', 'Similarity': ''}

    for item in recommended_items:
		# identificatore colonna S.xFilm
        sx_col = f'S.x {item}'

        # Totale S.xFilm
        sx_values = pd.to_numeric(df[sx_col], errors='coerce').fillna(0.0) #trasforma i valori vuoti prima in NaN poi in 0.0
        total = sx_values.sum() # somma dei valori
        total_row[item] = '' # lascia vuoto per la colonna del film
        total_row[sx_col] = round(total, 2) # assegna il totale arrotondato a 2 decimali

        # Somma delle similarità dei critici che hanno votato quel film
        mask = df[item] != '' # seleziona le righe dove il critico ha dato un voto al film
        sim_sum = df.loc[mask, 'Similarity'].astype(float).sum() # somma tutte le righe per cui il critico ha dato un voto
        sim_sum_row[item] = '' # lascia vuoto per la colonna del film
        sim_sum_row[sx_col] = round(sim_sum, 2) # assegna la somma delle similarità arrotondata a 2 decimali

        # Predizione: Total / Sim.Sum (solo se sim_sum != 0)
        if sim_sum != 0: 
            pred = prediction_function(person, item, total, sim_sum)
            pred_row[item] = ''
            pred_row[sx_col] = round(pred, 2)
        else:
            pred_row[item] = ''
            pred_row[sx_col] = ''

    ### Aggiunta delle tre righe al DataFrame ###
    df = pd.concat(
        [df, pd.DataFrame([total_row, sim_sum_row, pred_row])],
        ignore_index=True
    )

    return df

print("\nRecommendation table for Toby using Pearson Correlation:")	
print(create_recommendation_table('Toby', correlation=pearson_correlation, prediction_function=prediction_function1))

print("\nRecommendation table for Toby using Euclidean Similarity:")	
print(create_recommendation_table('Toby', correlation=euclidean_similarity, prediction_function=prediction_function1))

print("\nRecommendation table for Toby using another prediction function:")	
print(create_recommendation_table('Toby', correlation=pearson_correlation, prediction_function=prediction_function2))
