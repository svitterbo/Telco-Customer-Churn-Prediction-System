import pandas as pd

class DataManager:
    def __init__(self, ruta_archivo):
        self.ruta_archivo = ruta_archivo
        self.dataset = None
        self.X = None 
        self.y = None 

    def preparar_datos(self):
        try:
            self.dataset = pd.read_csv(self.ruta_archivo)
            
            if 'customerID' in self.dataset.columns:
                self.dataset = self.dataset.drop('customerID', axis=1)

            self.dataset['TotalCharges'] = pd.to_numeric(self.dataset['TotalCharges'], errors='coerce').fillna(0)

            
            self.y = self.dataset['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
            
            datos_sin_target = self.dataset.drop('Churn', axis=1)
            
            self.X = pd.get_dummies(datos_sin_target, drop_first=True)

            return True
            
        except FileNotFoundError:
            return False

    def obtener_datos_entrenamiento(self):
        """Devuelve los datos listos para el algoritmo."""
        return self.X, self.y