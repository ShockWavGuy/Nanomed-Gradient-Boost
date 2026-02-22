Project for the prediction of nanomedicine properties through gradient boosting and feature enrichment

Model Types Tested:
LMEM
LGBM
RF
CatBoost
GPBoost
LGBM + CatBoost
MLP

Main variance hypothesis is that heterogeneity, rather than batch effects, dominates signal; stronger biological & chemical terms may mitigate variance.
Mediocre/Poor MLP + GNN framework, as well as Catboost + group column categorical showed that batch effects hold minimal predictive power (or are near impossible to derive)


Datasets: 

Brain Delivery(https://www.cell.com/cell-biomaterials/fulltext/S3050-5623(25)00121-7)
                   PLGA Nanoparticle Formulations(https://www.nature.com/articles/s41597-025-05520-9)
