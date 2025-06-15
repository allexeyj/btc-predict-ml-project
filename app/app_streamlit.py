import streamlit as st
import asyncio

from service import ModelService


@st.cache_resource
def get_model_service():
    service = ModelService()
    asyncio.run(service.load_models())
    return service


service = get_model_service()

st.title("Прогноз Bitcoin")

model_list = service.list_models()
model_ids = [m["model_id"] for m in model_list]
model_descriptions = {m["model_id"]: m["description"] for m in model_list}

selected_model_id = st.selectbox(
    "Выберите модель из приведенного списка:",
    options=model_ids,
    format_func=lambda x: model_descriptions[x] if x in model_descriptions else x,
    index=model_ids.index(service.active_model_id) if service.active_model_id else 0,
)


if selected_model_id != service.active_model_id:
    if service.set_active_model(selected_model_id):
        st.success(f"Активирована модель: {selected_model_id}")
    else:
        st.error("Не удалось сменить модель")

text = st.text_area("Введите текст", height=200)
if st.button("Сделать предсказание"):
    if not text.strip():
        st.warning("Введите текст для предсказания")
    else:
        try:
            prediction = service.predict(text)
            st.success(f"Прогноз: {prediction:,.2f}")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
