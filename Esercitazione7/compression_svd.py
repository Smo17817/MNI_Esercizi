from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

# Valuta l'energia preservata nella SVD troncata per un dato k
def thumb_rule(S, k):
	energy = np.sum(S)
	energy_k = np.sum(S[:k])

	return energy_k / energy

# Facoltativo - Calcola il k necessario per preservare una certa soglia di energia
def thumb_rule_atK(S, energy_threshold):
	energy = np.sum(S) 
	cumulative_energy = 0
	k = 0
	while (cumulative_energy / energy) < energy_threshold:
		cumulative_energy += S[k]
		k += 1
	return k

# path directory di img
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(SCRIPT_DIR + "/img", "cane.png")

A = imread(IMG_PATH)

# A è una matrice  M x N x 3, essendo un'immagine RGB
# A(:,:,1) Red A(:,:,2) Blue A(:,:,3) Green
# su una scala tra 0 e 1
print(f"Risoluzione dell'immagine: {A.shape}")

X = np.mean(A,-1); # media lungo l'ultimo asse, cioè 2
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# If full_matrices=True (default), u and vT have the shapes (M, M) and (N, N), respectively.
# Otherwise, the shapes are (M, K) and (K, N), respectively, where K = min(M, N).
U, S, VT = np.linalg.svd(X,full_matrices=False)

#i valori di S precedenti, in particolare S(100), forniscono una stima dell'errore commesso
# nella compressione dell'immagine con la SVD
diag_S = np.diag(S)

j=0
for r in (5,20,100):
	Xapprox = U[:,:r] @ diag_S[0:r,:r] @ VT[:r,:]
	plt.figure(j+1)
	j +=1
	img = plt.imshow(Xapprox)
	img.set_cmap('gray')
	plt.axis('off')
	plt.title('r = ' + str(r))
	plt.show()
	print("Energia preservata per k={r}: ", thumb_rule(S,r))

# Facoltativo - ThumbRule@K con k = 80%
k = 0.8
print(f"\nK per il {k * 100}% di energia preservata: ", thumb_rule_atK(S, k))