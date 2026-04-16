import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ChurnModel:
    def __init__(self):
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.precision = 0
        self.entrenado = False

    def entrenar(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.modelo.fit(X_train, y_train)
        
        predicciones = self.modelo.predict(X_test)
        self.precision = accuracy_score(y_test, predicciones)
        self.entrenado = True
        
        return self.precision

    def predecir(self, datos_cliente):
        if self.entrenado:
            return self.modelo.predict_proba(datos_cliente)
        return None
        
    def obtener_importancias(self, columnas):
        if self.entrenado:
            importancias = self.modelo.feature_importances_
            df = pd.DataFrame({'Variable': columnas, 'Peso': importancias})
            return df.sort_values(by='Peso', ascending=False)
        return None