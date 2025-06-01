import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
from io import BytesIO

# إعداد الصفحة
st.set_page_config(
    page_title="تحليل المشاعر للنصوص العربية",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تحسين الأداء عن طريق التخزين المؤقت للنموذج
@st.cache_resource
def load_model():
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# تحسين مظهر الصفحة
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 2.5rem !important;
        text-align: center;
    }
    .uploadedFile {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("تحليل المشاعر للنصوص العربية")
st.write("ارفع ملف يحتوي على الجمل باللغة العربية، وسيتم تصنيف المشاعر إلى: إيجابية، سلبية، أو محايدة.")

# رفع الملف
uploaded_file = st.file_uploader("📁 رفع الملف", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # قراءة الملف مع تحسين التعامل مع الترميز
        if uploaded_file.name.endswith(".csv"):
            encodings = ['utf-8', 'ISO-8859-6', 'cp1256', 'utf-16']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"❌ خطأ في قراءة الملف: {str(e)}")
                    st.stop()
            if df is None:
                st.error("❌ لم نتمكن من قراءة الملف. الرجاء التأكد من الترميز.")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file)

        # التحقق من الأعمدة
        if "text" not in df.columns:
            st.error("❌ الملف يجب أن يحتوي على عمود باسم text.")
            st.stop()

        if df.empty or df["text"].isnull().all():
            st.error("❌ الملف فارغ أو لا يحتوي على بيانات نصية.")
            st.stop()

        # تحليل المشاعر مع شريط التقدم
        sentiment_pipeline = load_model()
        st.info("⏳ جاري تحليل المشاعر...")
        progress_bar = st.progress(0)
        
        # تحليل المشاعر مع تحديث شريط التقدم
        results = []
        total = len(df)
        for i, text in enumerate(df["text"].astype(str)):
            result = sentiment_pipeline(text)[0]
            results.append(result)
            progress_bar.progress((i + 1) / total)
        
        df["تصنيف المشاعر"] = [res["label"] for res in results]
        df["نسبة الثقة"] = [round(res["score"], 3) for res in results]

        # عرض النتائج
        st.success("✅ تم التحليل بنجاح!")
        
        # إضافة تصفية للنتائج
        sentiment_filter = st.selectbox(
            "تصفية النتائج حسب المشاعر:",
            ["الكل"] + list(df["تصنيف المشاعر"].unique())
        )
        
        filtered_df = df if sentiment_filter == "الكل" else df[df["تصنيف المشاعر"] == sentiment_filter]
        st.dataframe(filtered_df)

        # رسم بياني محسن
        st.markdown("### توزيع المشاعر:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_counts = df["تصنيف المشاعر"].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        bars = sentiment_counts.plot(
            kind='bar',
            ax=ax,
            color=[colors.get(x, '#3498db') for x in sentiment_counts.index]
        )
        
        # تحسين مظهر الرسم البياني
        plt.title("توزيع المشاعر في النصوص", pad=20)
        plt.ylabel("عدد النصوص")
        plt.xlabel("نوع المشاعر")
        plt.xticks(rotation=0)
        
        # إضافة القيم فوق الأعمدة
        for i, v in enumerate(sentiment_counts):
            ax.text(i, v, str(v), ha='center', va='bottom')
        
        st.pyplot(fig)

        # تصدير النتائج
        st.markdown("### تحميل النتائج:")
        
        # تحسين تصدير Excel
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="النتائج")
            workbook = writer.book
            worksheet = writer.sheets["النتائج"]
            
            # تنسيق الخلايا
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'bg_color': '#D9D9D9',
                'border': 1
            })
            
            # تنسيق البيانات
            data_format = workbook.add_format({
                'align': 'right',  # للنصوص العربية
                'border': 1
            })
            
            # تطبيق التنسيق
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                # تعيين عرض العمود
                max_length = max(
                    df[value].astype(str).apply(len).max(),
                    len(str(value))
                )
                worksheet.set_column(col_num, col_num, max_length + 2, data_format)

        excel_buffer.seek(0)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ تحميل النتائج (Excel)",
                data=excel_buffer.getvalue(),
                file_name="نتائج_تحليل.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="تحميل النتائج بتنسيق Excel مع دعم كامل للغة العربية"
            )

        # إحصائيات إضافية
        st.markdown("### إحصائيات التحليل:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("إجمالي النصوص", len(df))
        with col2:
            positive_percentage = round((df["تصنيف المشاعر"] == "positive").mean() * 100, 1)
            st.metric("نسبة المشاعر الإيجابية", f"{positive_percentage}%")
        with col3:
            avg_confidence = round(df["نسبة الثقة"].mean() * 100, 1)
            st.metric("متوسط نسبة الثقة", f"{avg_confidence}%")

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحليل الملف: {str(e)}")

# توقيع شخصي محسن
st.markdown("""---""")
st.markdown("""
<div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
    <h4 style="color: #2c3e50; margin-bottom: 0.5rem;">تم التطوير بواسطة</h4>
    <p style="color: #34495e;">
        📧 <a href="mailto:Waseeme900@gmail.com">Waseeme900@gmail.com</a> | 
        🔗 <a href="https://www.linkedin.com/in/waseemalmazrua" target="_blank">LinkedIn</a>
    </p>
    <p style="color: #7f8c8d; font-size: 0.8rem;">© وسيم المزروع - تطبيق تحليل المشاعر</p>
</div>
""", unsafe_allow_html=True)
