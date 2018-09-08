# Lik-ML
API Machine learning

En travaux! Mais pour bientôt!

But : J'ai fais cette api dans un but éducatif, comprendre les algorithm qui se cache derrière le machine learning. L'optimisation est la bienvenue quand bien même elle n'est pas le but ultime de cette API.

Fonctionnement : Cette API fonctionne sous le principe de layer(couche) et de module, un module peut contenir plusieur layer agencer différemment en fonction du type de module choisie. Les layer sont les élément de calcule, il vont contenir les variable ainsie que les fonction.

Layer:

FullyConnected : Un layer qui connecte chaque entrée a chaque sortie, avec un poid associé a chaque connection.
Loss : Un layer qui ce place en sortie, elle calcule la perte entre la sortie et l'exemple, fait aussi office de FullyConnected.
Convolution* : Un layer qui fait une convolution, avec un poid associer a chaque valeur du kernel de convolution.
Module*:

Sequential : Les layer sont connecter les un a la suit des autre.
fonction d'activation:

ACTIVATION_LINEAR
ACTIVATION_SIGMOIDE
ACTIVATION_SOFT_SIGN
ACTIVATION_RELU
ACTIVATION_GAUSSIAN
ACTIVATION_SILU
fonction Loss:

SQUAREDLOSS
LOGLOSS
CROSSENTROPY
NEGATIVELOGLIKELIHOOD*
'*' pas encore implémenté
