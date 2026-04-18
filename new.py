import google.generativeai as genai
import os

# 1. حطي مفتاح الـ API تبعك هون
API_KEY = "AIzaSyDqb9borRAuQI9P034aRZNzXD2bfAw6FV8"
genai.configure(api_key=API_KEY)

def convert_to_gloss(arabic_text):
    # 2. إعداد النموذج (يفضل استخدام gemini-1.5-flash للسرعة والتكلفة أو pro للدقة)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 3. تصميم الـ Prompt (هاد الجزء الأهم)
    # ملاحظة: بما إنك بنيتي الداتا أصلاً بجيميناي، الـ Prompt لازم يكون مشابه للي استخدمتيه بالبناء
    prompt = f"""
    حول النص الطبي العربي التالي إلى Gloss (لغة إشارة عربية).
    القواعد:
    - استخرج اسم الدواء، الجرعة، والوقت.
    - الترتيب: [أنت/مريض] + [اسم الدواء] + [الجرعة] + [الفعل] + [الوقت/التكرار].
    - استخدم كلمات بسيطة ودقيقة.
    
    النص: "{arabic_text}"
    الـ Gloss:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"خطأ في الاتصال: {str(e)}"

# مثال للاختبار
text = "تناول حبة بنادول 500 ملغ كل 8 ساعات"
print(f"Gloss: {convert_to_gloss(text)}")
