import re
import json
import ast
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from data import samples


MODEL_NAME = "google/mt5-small"


TARGET_SCHEMA = {
    "drug_name": None,
    "indication": None,
    "dose_value": None,
    "dose_unit": None,
    "dosage_form": None,
    "frequency_text": None,
    "times_per_day": None,
    "interval_hours": None,
    "timing": None,
    "duration_value": None,
    "duration_unit": None,
    "route": None,
    "prn": False,
    "max_daily_dose": None,
    "warnings": [],
    "contraindications": [],
    "notes": []
}


AR_NUMBERS = {
    "صفر": 0,
    "واحد": 1,
    "واحدة": 1,
    "اثنين": 2,
    "اتنين": 2,
    "مرتين": 2,
    "نقطتين": 2,
    "ثلاثة": 3,
    "تلاتة": 3,
    "تلات": 3,
    "ثلاث": 3,
    "أربعة": 4,
    "اربعة": 4,
    "أربع": 4,
    "اربع": 4,
    "خمسة": 5,
    "خمس": 5,
    "سبعة": 7,
    "سبع": 7
}


def normalize_arabic_text(text):
    text = text.strip()
    text = text.replace("الأكل", "الاكل")
    text = text.replace("الفطور", "الافطار")
    text = text.replace("عالريق", "على الريق")
    text = text.replace("تيام", "ايام")
    text = text.replace("تلات", "ثلاث")
    text = text.replace("مسا", "مساء")
    text = text.replace("الادن", "الاذن")
    text = text.replace("إقياء", "اقياء")
    text = re.sub(r"\s+", " ", text)
    return text


def build_prompt(text):
    schema_str = json.dumps(TARGET_SCHEMA, ensure_ascii=False)
    prompt = f"""
استخرج المعلومات الدوائية من النص العربي التالي.
أعد النتيجة فقط بصيغة JSON صحيحة وبدون أي شرح.

الحقول المطلوبة:
{schema_str}

تعليمات:
- إذا لم تعرف القيمة ضع null
- warnings و contraindications و notes يجب أن تكون قوائم
- prn يجب أن تكون true أو false
- لا تضف أي نص خارج JSON

النص:
{text}
"""
    return prompt.strip()


def safe_json_parse(text):
    text = text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return None


def extract_number_from_text(text):
    digit_match = re.search(r"\b(\d+)\b", text)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in AR_NUMBERS.items():
        if word in text:
            return value
    return None


def detect_frequency(text):
    text = normalize_arabic_text(text)

    if "كل 12 ساعة" in text:
        return {"frequency_text": "كل 12 ساعة", "times_per_day": 2, "interval_hours": 12}
    if "كل 8 ساعات" in text:
        return {"frequency_text": "كل 8 ساعات", "times_per_day": 3, "interval_hours": 8}
    if "مرتين يومياً" in text or "مرتين يوميا" in text or "صبح ومساء" in text:
        return {"frequency_text": "مرتين يومياً", "times_per_day": 2, "interval_hours": 12}
    if "3 مرات باليوم" in text or "ثلاث مرات باليوم" in text or "لحد ثلاث مرات باليوم" in text:
        return {"frequency_text": "ثلاث مرات يومياً", "times_per_day": 3, "interval_hours": 8}
    if "كل يوم" in text or "مرة يومياً" in text or "مرة يوميا" in text:
        return {"frequency_text": "مرة يومياً", "times_per_day": 1, "interval_hours": 24}
    if "بالاسبوع" in text or "بالأسبوع" in text:
        return {"frequency_text": "مرة أسبوعياً", "times_per_day": None, "interval_hours": None}

    return {"frequency_text": None, "times_per_day": None, "interval_hours": None}


def detect_duration(text):
    text = normalize_arabic_text(text)

    if "لسبع ايام" in text or "لسبعة ايام" in text:
        return {"duration_value": 7, "duration_unit": "يوم"}
    if "لثلاث ايام" in text or "لثلاثة ايام" in text:
        return {"duration_value": 3, "duration_unit": "يوم"}
    if "خلال يوم ل ثلاث ايام" in text or "خلال يوم لثلاث ايام" in text:
        return {"duration_value": 3, "duration_unit": "يوم تقريباً"}
    if "بالاسبوع" in text or "بالأسبوع" in text:
        return {"duration_value": 1, "duration_unit": "أسبوع"}

    return {"duration_value": None, "duration_unit": None}


def detect_timing(text):
    text = normalize_arabic_text(text)
    found = []

    timing_rules = [
        ("بعد الاكل", "بعد الأكل"),
        ("قبل الاكل", "قبل الأكل"),
        ("قبل الاكل بنص ساعة", "قبل الأكل بنصف ساعة"),
        ("قبل الفطور", "قبل الفطور"),
        ("على الريق", "على الريق"),
        ("الصبح", "صباحاً"),
        ("صباح", "صباحاً"),
        ("مساء", "مساءً"),
        ("بعد الفطور", "بعد الفطور"),
    ]

    for phrase, label in timing_rules:
        if phrase in text:
            found.append(label)

    if not found:
        return None

    return " + ".join(list(dict.fromkeys(found)))


def detect_dose_and_form(text):
    text = normalize_arabic_text(text)

    dose_value = None
    dose_unit = None
    dosage_form = None
    route = None

    if "نقطتين" in text:
        dose_value = 2
        dose_unit = "نقطة"
        dosage_form = "قطرة"
        route = "أذن"
    elif "معلقة كبيرة" in text or "ملعقة كبيرة" in text:
        dose_value = 1
        dose_unit = "ملعقة كبيرة"
        dosage_form = "شراب"
        route = "فموي"
    elif "حبة" in text:
        dose_value = 1
        dose_unit = "حبة"
        dosage_form = "حبوب"
        route = "فموي"

    if "شراب" in text:
        dosage_form = "شراب"
    if "قطرة" in text:
        dosage_form = "قطرة"

    return {
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "dosage_form": dosage_form,
        "route": route
    }


def detect_indication_and_drug(text):
    text = normalize_arabic_text(text)

    drug_name = None
    indication = None
    contraindications = []
    warnings = []
    notes = []

    if "سيبروكورت" in text:
        drug_name = "سيبروكورت"
    if "فلام كي" in text:
        drug_name = "فلام كي"

    if "واقي معدة" in text:
        indication = "واقي معدة"
    elif "للسعلة" in text or "دوا سعلة" in text or "السعلة" in text:
        indication = "سعال"
    elif "مضاد اقياء" in text:
        indication = "مضاد إقياء"
    elif "التهاب الاذن" in text:
        indication = "التهاب الأذن"
    elif "لوجع الاسنان" in text:
        indication = "ألم الأسنان"

    if "حساسية" in text:
        warnings.append("تحقق من الحساسية")
    if "ضغط" in text:
        warnings.append("تحذير لمرضى الضغط")
    if "بعد الاكل" in text:
        warnings.append("يؤخذ بعد الأكل لتخفيف تهيج المعدة")
    if "وقفو" in text:
        warnings.append("إيقاف دواء أو مكمل متزامن")
    if "يرفع الضغط" in text:
        contraindications.append("قد يرفع الضغط")
    if "خال من السكر" in text or "ما فيو سكر" in text:
        notes.append("خال من السكر")
    if "عشبة لبلاب" in text:
        notes.append("مستحضر عشبي - لبلاب")

    return {
        "drug_name": drug_name,
        "indication": indication,
        "warnings": list(dict.fromkeys(warnings)),
        "contraindications": list(dict.fromkeys(contraindications)),
        "notes": list(dict.fromkeys(notes))
    }


def empty_schema():
    return {
        "drug_name": None,
        "indication": None,
        "dose_value": None,
        "dose_unit": None,
        "dosage_form": None,
        "frequency_text": None,
        "times_per_day": None,
        "interval_hours": None,
        "timing": None,
        "duration_value": None,
        "duration_unit": None,
        "route": None,
        "prn": False,
        "max_daily_dose": None,
        "warnings": [],
        "contraindications": [],
        "notes": []
    }


def expert_system_extract(text):
    text_norm = normalize_arabic_text(text)

    result = {}
    result.update(detect_dose_and_form(text_norm))
    result.update(detect_frequency(text_norm))
    result.update(detect_duration(text_norm))
    result["timing"] = detect_timing(text_norm)
    result.update(detect_indication_and_drug(text_norm))

    result["prn"] = "عند اللزوم" in text_norm

    if "لا تتجاوز 4" in text_norm or "أربع جرعات" in text_norm or "اربع جرعات" in text_norm:
        result["max_daily_dose"] = 4
    else:
        result["max_daily_dose"] = None

    return result


def merge_outputs(model_data, rule_data):
    final = empty_schema()

    if isinstance(model_data, dict):
        for key in final.keys():
            if key in model_data and model_data[key] not in [None, "", []]:
                final[key] = model_data[key]

    for key, value in rule_data.items():
        if key in ["warnings", "contraindications", "notes"]:
            merged = (final.get(key) or []) + value
            final[key] = list(dict.fromkeys([x for x in merged if x]))
        else:
            if value not in [None, "", []]:
                final[key] = value

    return final


def validate_and_flag(slots, original_text):
    flags = []

    if not slots["dose_value"]:
        flags.append("نقص: الجرعة غير واضحة")

    if not slots["frequency_text"]:
        flags.append("نقص: التكرار غير واضح")

    if "3 مرات باليوم" in original_text and slots["times_per_day"] == 3 and slots["interval_hours"] not in [None, 8]:
        flags.append("تعارض: التكرار لا يطابق الفاصل الزمني")

    if slots["dosage_form"] == "شراب" and slots["dose_unit"] == "حبة":
        flags.append("تعارض: الشراب لا يطابق وحدة حبة")

    if "قطرة" in original_text and slots["route"] is None:
        flags.append("ملاحظة: طريق الإعطاء غير محدد")

    return slots, flags


def build_structured_arabic_output(slots):
    parts = []

    if slots["indication"]:
        parts.append(f"الاستطباب: {slots['indication']}")
    if slots["drug_name"]:
        parts.append(f"اسم الدواء: {slots['drug_name']}")
    if slots["dose_value"] and slots["dose_unit"]:
        parts.append(f"الجرعة: {slots['dose_value']} {slots['dose_unit']}")
    if slots["dosage_form"]:
        parts.append(f"الشكل الدوائي: {slots['dosage_form']}")
    if slots["frequency_text"]:
        parts.append(f"التكرار: {slots['frequency_text']}")
    if slots["timing"]:
        parts.append(f"التوقيت: {slots['timing']}")
    if slots["duration_value"] and slots["duration_unit"]:
        parts.append(f"المدة: {slots['duration_value']} {slots['duration_unit']}")
    if slots["route"]:
        parts.append(f"طريق الإعطاء: {slots['route']}")
    if slots["max_daily_dose"]:
        parts.append(f"الحد الأعلى اليومي: {slots['max_daily_dose']}")
    if slots["warnings"]:
        parts.append("التحذيرات: " + " | ".join(slots["warnings"]))
    if slots["contraindications"]:
        parts.append("موانع/احتياطات: " + " | ".join(slots["contraindications"]))
    if slots["notes"]:
        parts.append("ملاحظات: " + " | ".join(slots["notes"]))

    return " ؛ ".join(parts)


def transformer_extract(tokenizer, model, text, max_new_tokens=256):
    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    parsed = safe_json_parse(output)

    return {
        "raw_model_output": output,
        "parsed": parsed
    }


def main():
    print("تحميل نموذج Transformer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    rows = []

    for idx, text in enumerate(samples, start=1):
        model_out = transformer_extract(tokenizer, model, text)        
        rules_out = expert_system_extract(text)
        merged = merge_outputs(model_out["parsed"], rules_out)
        merged, flags = validate_and_flag(merged, text)
        final_text = build_structured_arabic_output(merged)

        rows.append({
            "id": idx,
            "input_text": text,
            "raw_model_output": model_out["raw_model_output"],
            "final_slots": json.dumps(merged, ensure_ascii=False),
            "flags": " | ".join(flags) if flags else "",
            "final_structured_text": final_text
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("النتائج النهائية")
    print("=" * 120)

    for _, row in df.iterrows():
        print("\n" + "=" * 120)
        print(f"ID: {row['id']}")
        print(f"النص الأصلي:\n{row['input_text']}\n")
        print(f"الناتج النهائي:\n{row['final_structured_text']}\n")
        print(f"Flags:\n{row['flags'] if row['flags'] else 'لا يوجد'}\n")

    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print("تم حفظ النتائج في ملف results.csv")


if __name__ == "__main__":
    main()