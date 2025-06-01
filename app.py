import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from io import BytesIO

# إعداد الصفحة
st.set_page_config(page_title="تحليل المشاعر للنصوص العربية", layout="centered")
st.title(" تحليل المشاعر للنصوص العربية")
st.write("ارفع ملف يحتوي على الجمل باللغة العربية، وسيتم تصنيف المشاعر إلى: إيجابية، سلبية، أو محايدة.")

# رفع الملف
uploaded_file = st.file_uploader("📁 رفع الملف", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # قراءة الملف
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-6')
        else:
            df = pd.read_excel(uploaded_file)

        # التحقق من الأعمدة
        if "text" not in df.columns:
            st.error("❌ الملف يجب أن يحتوي على عمود باسم text.")
            st.stop()

        if df.empty or df["text"].isnull().all():
            st.error("❌ الملف فارغ أو لا يحتوي على بيانات نصية.")
            st.stop()

        # تحميل النموذج العربي من Hugging Face
        model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # تحليل المشاعر
        st.info("⏳ جاري تحليل المشاعر...")
        results = sentiment_pipeline(df["text"].astype(str).tolist())
        df["تصنيف المشاعر"] = [res["label"] for res in results]
        df["نسبة الثقة"] = [round(res["score"], 3) for res in results]

        # عرض النتائج
        st.success("✅ تم التحليل بنجاح، النتائج أدناه:")
        st.dataframe(df)

        # رسم بياني (labels بالإنجليزية عشان الخط العربي)
        st.markdown("###  Sentiment Distribution:")
        fig, ax = plt.subplots()
        df["تصنيف المشاعر"].value_counts().plot(kind='bar', ax=ax, color=['green', 'red', 'gray'])
        ax.set_ylabel("Count")
        ax.set_xlabel("Sentiment")
        plt.xticks(rotation=0)
        st.pyplot(fig)

        # زر تحميل CSV
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ تحميل النتائج بصيغة CSV", data=csv, file_name="نتائج_تحليل.csv", mime="text/csv")

        excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name="النتائج")
excel_buffer.seek(0)  # إضافة مهمة للتأكد من قراءة الملف من البداية

st.download_button("⬇️ تحميل النتائج بصيغة Excel", data=excel_buffer.getvalue(),
                   file_name="نتائج_تحليل.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل الملف: {e}")
# توقيع شخصي بسيط في أسفل الصفحة
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; font-size: 14px; color: gray;">
📧 Waseeme900@gmail.com  |  
🔗 <a href="https://www.linkedin.com/in/waseemalmazrua" target="_blank">LinkedIn</a><br>
© وسيم المزروع - Sentiment Analysis App
</div>
""", unsafe_allow_html=True)
