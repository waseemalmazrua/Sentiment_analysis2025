import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt

# إعداد الصفحة
st.set_page_config(page_title="تحليل المشاعر", layout="centered")
st.title(" تحليل المشاعر للنصوص العربية")
st.write("ارفع ملف يحتوي على الجمل التي ترغب بتحليل مشاعرها، وسيتم تصنيفها تلقائيًا إلى إيجابية، سلبية، أو محايدة.")

# رفع الملف
uploaded_file = st.file_uploader(" رفع الملف", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ الملف لا يحتوي على عمود باسم text.")
        else:
            model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

            المشاعر = []
            الثقة = []

            for نص in df["text"]:
                نتيجة = sentiment_pipeline(str(نص))[0]
                المشاعر.append(نتيجة['label'])
                الثقة.append(round(نتيجة['score'], 3))

            df["تصنيف المشاعر"] = المشاعر
            df["نسبة الثقة"] = الثقة

            st.success("✅ تم التحليل بنجاح:")
            st.dataframe(df)

            st.markdown("###  توزيع المشاعر:")
            fig, ax = plt.subplots()
            df["تصنيف المشاعر"].value_counts().plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
            ax.set_ylabel("عدد الجمل")
            ax.set_xlabel("نوع المشاعر")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"حدث خطأ أثناء تحليل الملف: {e}")
