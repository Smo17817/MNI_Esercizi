import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla

# ## Problema
# 4 utenti e loro valutazioni per 6 stagioni
# 
# | Season   | Ryne | Erin  | Nathan | Pete  |
# | season 1 | 5    | 5     | 0      | 5     |
# | season 2 | 5    | 0     | 3      | 4     |
# | season 3 | 3    | 4     | 0      | 3     |
# | season 4 | 0    | 0     | 5      | 3     |
# | season 5 | 5    | 4     | 4      | 5     |
# | season 6 | 5    | 4     | 5      | 5     |

k = 2

seasons = [1,2,3,4,5,6]
users = ['Ryne', 'Erin', 'Nathan', 'Pete']

A = np.array([            
    [5,5,0,5], # season 1
    [5,0,3,4], # season 2
    [3,4,0,3], # season 3
    [0,0,5,3], # season 4
    [5,4,4,5], # season 5
    [5,4,5,5]  # season 6
    ], dtype=float)

### Calcoliamo la SVD di A
print('Dimensions:')
U, s, VT = sla.svd(A)
V = VT.T
print(f"U: {U.shape}")      # Matrice delle stagioni (item)
print(f"s: {s.shape}")      # Vettore dei valori singolari
print(f"VT: {VT.shape}")    # Matrice degli utenti
print('Vector of Singular Values: ', s)

### rappresentiamo i dati in uno spazio a k dimensioni
U_k = U[:,:k]               # troncamento delle prime k colonne di U
V_k = V[:,:k]               # troncamento delle prime k righe di V
S_k = np.diag(s[:k])        # matrice diagonale con i primi k valori singolari
print(f'SVD troncata con k={k}. Stampa di U, s e V con 2 cifre decimali')
print(f'\nU_{k}:\n', U_k.round(2))
print(f'\nS_{k}:\n', S_k.round(2))
print(f'\nV_{k}:\n', V_k.round(2))

### Grafichiamo la proiezione dei dati in uno spazio 2D
# Le prime k colonne di U rappresentano le stagioni - Sfere blu
# Le prime k righe di V rappresentano gli utenti - Quadrati rossi.
# x rappresenti la prima componente, y la seconda

plt.plot(U_k[:,0], U_k[:,1], 'bo', markersize=15, clip_on=False, label='seasons')
plt.plot(V_k[:,0], V_k[:,1], 'rs', markersize=15, clip_on=False, label='users')

ax = plt.gca()
for i, txt in enumerate(seasons):
    ax.text(U_k[i,0], U_k[i,1], txt, ha='left', va='bottom', fontsize=20)
    
for i, txt in enumerate(users):
    ax.text(V_k[i,0], V_k[i,1], txt, ha='left', va='bottom', fontsize=20)

# axis trickery - aggiusta gli assi per farli intersecare al centro
ax = plt.gca()
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

### Obiettivo: trovare utenti simili per fornire raccomandazioni
# Aggiungiamo l'utente Luke, che dà le valutazioni seguenti per le stagioni: [5,5,0,0,0,5].
# Per fornire raccomandazioni a Luke troviamo gli utenti più simili a Luke,
# rappresentando Luke nello spazio 2D in cui abbiamo rappresentato gli utenti

luke = np.array([5,5,0,0,0,5])
print('\nValutazioni delle stagioni di Luke', luke)

# Per proiettare le valutazioni di Luke nello spazio 2D degli utenti, se esse sono indicate con L
# L^T * U_2 * S_2^{-1}

luke2d = luke.dot(U_k.dot(np.linalg.inv(S_k)))
print(f'Valutazioni di Luke proiettate nello spazio 2D: {luke2d}')

# Grafichiamo le valutazioni di Luke così rappresentate
plt.plot(U_k[:,0], U_k[:,1], 'bo', markersize=15, clip_on=False, label='seasons')
plt.plot(V_k[:,0], V_k[:,1], 'rs', markersize=15, clip_on=False, label='users')

ax = plt.gca()
for i, txt in enumerate(seasons):
    ax.text(U_k[i,0], U_k[i,1], txt, ha='left', va='bottom', fontsize=20)
    
for i, txt in enumerate(users):
    ax.text(V_k[i,0], V_k[i,1], txt, ha='left', va='bottom', fontsize=20)

# Viene aggiunto Luke al grafico - Stella verde
plt.plot(luke2d[0], luke2d[1], 'g*', markersize=15, clip_on=False, label='luke')
ax.text(luke2d[0], luke2d[1], 'Luke', ha='left', va='bottom', fontsize=20)

# axis trickery - aggiusta gli assi per farli intersecare al centro
ax = plt.gca()
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('right')

# Osserviamo che gli angoli minori sono tra Luke e Pete e tra Luke e Ryne
# per quantificare la distanza usiamo la similarità coseno
# similarity (a,b) = (a,b) / (||a|| ||b||)

print(f"Dimensioni di Luke proiettate nello spazio k-dimensionale: {luke2d.shape}")

# calcoliamo la similarità coseno tra Luke e ciascun utente
sim_dict = {}

for i, xy in enumerate(V_k):
    angle = np.dot(xy, luke2d) / (np.linalg.norm(xy) * np.linalg.norm(luke2d))
    # Aggiungo ogni similarità calcolata al dizionario
    sim_dict[users[i]] = angle

sorted_keys = sorted(sim_dict, key=sim_dict.get, reverse=True)

print('\nUtenti più simili a Luke in ordine decrescente di similarità coseno:')
for key in sorted_keys:
    print(f'{key}: {sim_dict[key].round(2)}')

if k < 3:
    plt.show()
