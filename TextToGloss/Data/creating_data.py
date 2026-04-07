# medical_gloss_engine.py

import re

# 1. Stopwords
STOPWORDS = {
    'في','على','من','إلى','عن','مع','و','أو','ثم','هذا','هذه','ذلك','هناك','هنا'
}

# 2. كلمات طبية لا تُحذف
MEDICAL_TERMS = {
    'السكري','الضغط','الربو','القلب','الكبد','الكلى','المعدة'
}

# 3. كلمات حرجة
CRITICAL = {'قبل','بعد','لا','ألم','جرعة','دواء','حقن'}

# 4. تحويل عامي → فصحى
DIALECT_MAP = {
    'فيو': 'يوجد',
    'مافي': 'لا يوجد',
    'بدو': 'يريد',
    'عم': '',
    'ياخد': 'يأخذ',
    'وجعو': 'ألم'
}

# 5. تبسيط الأفعال
VERB_MAP = {
    'يتناول': 'أخذ',
    'يستخدم': 'استعمال',
    'يعاني': 'ألم',
    'يشكو': 'ألم'
}

# 6. الزمن
TIME_WORDS = {
    'صباحاً': 'صباح',
    'مساءً': 'مساء',
    'الآن': 'الآن',
    'سابقاً': 'قبل'
}

def normalize_dialect(text):
    for k, v in DIALECT_MAP.items():
        text = text.replace(k, v)
    return text

def remove_al(word):
    if word.startswith("ال") and word not in MEDICAL_TERMS:
        return word[2:]
    return word

def simplify_verb(word):
    return VERB_MAP.get(word, word)

def normalize_time(word):
    return TIME_WORDS.get(word, word)

def tokenize(text):
    return re.findall(r'\w+', text)

def filter_words(words):
    result = []
    for w in words:
        if w in STOPWORDS and w not in CRITICAL:
            continue
        result.append(w)
    return result

def process_sentence(text):
    text = normalize_dialect(text)

    words = tokenize(text)

    words = [normalize_time(w) for w in words]
    words = [remove_al(w) for w in words]
    words = [simplify_verb(w) for w in words]
    words = filter_words(words)

    # إعادة ترتيب (بدائي)
    time = [w for w in words if w in TIME_WORDS.values()]
    rest = [w for w in words if w not in TIME_WORDS.values()]

    return " ".join(time + rest)

# تجربة
if __name__ == "__main__":
    while True:
        text = input("أدخل جملة طبية: ")
        print("Gloss:", process_sentence(text))