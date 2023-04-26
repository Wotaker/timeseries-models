# timeseries-models
A Project for the Data Mining course in the AGH 2023 Datascience class. We compare a few recent models in the job of timeseries data prediction

## Cele projektu
Zastosowanie algorytmów Gradient Boosted Decision Trees / Light Gradient Boosting do prognozowania szeregów czasowych (biblioteka XGBoost)
* Przykładowe zbiory danych:
  * Przewidywanie kursu wybranych kryptowalut
  * [Store Sales Forecasting](https://www.kaggle.com/c/walmartrecruiting-store-sales-forecasting/data)
  * [**Sunspots**](https://www.kaggle.com/datasets/robervalt/sunspots) **<- WYBRANO**
* Dopasowywanie parametrów modelu
* Porównanie z modelami: Arima, ESN, Prophet

## Devops
### Środowisko
Aby korzystac z projektu w formie biblioteki, nalezy z root projektu `timeseries-models/` poleceniem  
`pip install -e .`  
zainstalowac projekt w formie pakietu. Wymagane requirements powinny się zainstalowac automatycznie. Polecane jest uprzednie stworzenie środowiska wirtualnego poleceniem  
`python -m venv venv`
### Wersjonowanie
Jako ze kazdy z nas skupia się na innym modelu, proponuję pracowac w osobnych branchach, np od 1 litery imienia + nazwisko:  
  `git branch -b wciezobka`.