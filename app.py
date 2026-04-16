import streamlit as st
from data_manager import DataManager
from model import ChurnModel
import random

st.set_page_config(page_title="Predicción de Fuga", page_icon="📊", layout="wide")

st.title("📊 Sistema de Predicción de Churn Rate")
st.markdown("---")

if 'cerebro' not in st.session_state:
    st.session_state.cerebro = ChurnModel()
    st.session_state.gestor = DataManager("telco_churn.csv")
    st.session_state.entrenado = False

gestor = st.session_state.gestor
cerebro = st.session_state.cerebro

if gestor.preparar_datos():
    X, y = gestor.obtener_datos_entrenamiento()
    
    st.header("🧠 Entrenamiento de la IA")
    if st.button("Entrenar Modelo", type="primary"):
        with st.spinner("La Inteligencia Artificial está estudiando..."):
            precision = cerebro.entrenar(X, y)
            st.session_state.entrenado = True
            
        st.success("¡Entrenamiento finalizado!")
        st.metric(label="Precisión del Modelo", value=f"{precision * 100:.2f}%")

    if st.session_state.entrenado:
        
        st.markdown("---")
        st.header("🔬 ¿Por qué se van los clientes?")
        st.write("Esta es la radiografía del Random Forest. Muestra qué datos tienen más peso a la hora de predecir la fuga.")
        
        importancias = cerebro.obtener_importancias(X.columns)
        
        st.bar_chart(data=importancias.head(10).set_index('Variable'))
        
        st.info("💡 **Dato de negocio:** Si te fijás, la guita (TotalCharges/MonthlyCharges) y la antigüedad (tenure) son las que mandan. Pero ojo con el tipo de contrato, suele ser clave.")

        st.markdown("---")
        st.header("🔮 Predicción en Vivo")
        
        if st.button("Analizar Cliente al Azar"):
            indice = random.randint(0, len(X) - 1)
            cliente_datos = X.iloc[[indice]]
            cliente_realidad = y.iloc[indice]
            
            probabilidad = cerebro.predecir(cliente_datos)
            prob_fuga = probabilidad[0][1] * 100 
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Datos del Cliente")
                st.dataframe(cliente_datos.T) 
                
            with col2:
                st.subheader("Veredicto del Algoritmo")
                if prob_fuga > 50:
                    st.error(f"🚨 ALTO RIESGO: {prob_fuga:.1f}% de probabilidad de fuga.")
                else:
                    st.success(f"✅ CLIENTE FIDELIZADO: Solo {prob_fuga:.1f}% de riesgo.")
                    
                st.markdown("---")
                texto_realidad = "Se dio de baja ❌" if cliente_realidad == 1 else "Se quedó ✅"
                st.write(f"**Dato histórico real:** {texto_realidad}")

else:
    st.error("Error: No se encontró el archivo 'telco_churn.csv'.")