import google.generativeai as genai
import pandas as pd
import time

# 1. إعداد مفتاح الواجهة (استبدل 'YOUR_API_KEY' بمفتاحك الخاص)
genai.configure(api_key="AIzaSyDqb9borRAuQI9P034aRZNzXD2bfAw6FV8")
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_sentences(gloss):
    prompt = f"قم بتوليد 20 جملة عربية قصيرة وبسيطة تحتوي على الكلمة (Gloss): '{gloss}'. أريد النتائج كقائمة مرقمة فقط."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"خطأ في معالجة {gloss}: {e}")
        return None

# 2. تحميل ملف الـ CSV
input_file = 'TextToGloss.xlsx'  # تأكد أن اسم العمود في الملف هو 'gloss'
df = pd.read_csv(input_file)

# 3. معالجة البيانات
results = []

print(f"بدء التوليد لـ {len(df)} كلمة...")

for index, row in df.iterrows():
    gloss = row['gloss']
    print(f"يتم الآن معالجة: {gloss} ({index + 1}/{len(df)})")
    
    sentences = generate_sentences(gloss)
    
    if sentences:
        results.append({'gloss': gloss, 'generated_sentences': sentences})
    
    # تأخير بسيط لتجنب تخطي حدود الاستخدام المجاني (Rate Limit)
    time.sleep(2) 

# 4. حفظ النتائج في ملف جديد
output_df = pd.DataFrame(results)
output_df.to_csv('generated_dataset.csv', index=False, encoding='utf-8-sig')

print("تم الانتهاء! تفقد ملف generated_dataset.csv")