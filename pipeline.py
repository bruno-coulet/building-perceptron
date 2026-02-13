from sklearn import Pipeline

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.metrics import classification_report, confusion_matrix



import seaborn as sns, matplotlib.pyplot as plt



# Définition des models Regressors - dict {nom : tuples ( Modèle, dict{Grille_Hyperparamètres} ) }
regressors = {
    'LinearRegression': (LinearRegression(), 
                        {                       
                        'regressor__fit_intercept': [True, False],
                        'regressor__positive': [True, False]
                        }),

    'KNeighborsRegressor': (KNeighborsRegressor(),
                            {
                            'regressor__n_neighbors': [x for x in range (5, 11)],
                            'regressor__weights': ['uniform', 'distance'],
                            'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'regressor__metric': ['euclidean', 'manhattan', 'minkowski']
                            }),

    'XGBRegressor': (XGBRegressor(),
                    {
                    'regressor__n_estimators': [100, 150, 200], 
                    'regressor__learning_rate': [0.01, 0.1, 0.2], 
                    'regressor__max_depth': [3, 4, 5], 
                    'regressor__subsample': [0.6, 0.8, 1.0], 
                    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
                    })
}


# Collecte des scores
results = {}

# Boucle sur chaque algo avec pipeline d'entraînement
for name, (reg_model, param_grid) in regressors.items():
    print(f"Recherche pour {name}...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),    # Standardscaler
        ('regressor', reg_model)         # puis test de l'algo
    ])

    # Définition et entrainement du GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='r2'
    )

    grid_search.fit(X_train_processed, y_train)

    # Récupération des meilleurs paramètres
    best_pipeline = grid_search.best_estimator_

    # Calcul du R2 sur le test set
    y_pred_test = best_pipeline.predict(X_test_processed)
    r2_test = r2_score(y_test, y_pred_test)

    # Calcul de la RMSE et MAE sur le test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_tet = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Incrémentation du dictionnaire de résultats
    results[name] = {
        'R2_train': cross_val_score(
            best_pipeline,
            X-train_processed,
            y_train, cv=5, scoring='r2' ),
            
        'R2_test': r2_test, 
        'RMSE_test': rmse_tet, 
        'MAE_test': mae_test
        }


# Visualisation des résultats

colors = ['blue', 'orange', 'green']
palette = dict(zip(regressors.keys(), colors))

sns.boxplot(
    data=[results[name]['R2_train'] for name in regressors.keys()],
    ax=ax,
    palette=colors
)

ax.set_xticks(range(len(regressors)))
ax.set_xticklabels(regressors.keys())
ax.set_title('Variabilité du R2 train poue chaquea lgo avec GrisSearchCV\n')
ax.set_ylabel('R2 train')
plt.show()





  












    pipeline = Pipeline([
        ('scaler', StandardScaler()),    # Standardscaler
        ('classifier', algo)            # puis test de l'algo
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_

    y_pred = best_pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

               