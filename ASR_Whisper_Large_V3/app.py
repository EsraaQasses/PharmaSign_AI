import re
import torch, torchaudio, soundfile as sf
import numpy as np
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = re.compile(r"\u0640")
EXTRA_SPACES = re.compile(r"\s+")
PUNCT_SPACES = re.compile(r"\s*([،؛؟,.!])\s*")

AR_INDIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
FA_INDIC_TO_LATIN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

WORD_MAP_STRONG = {
    "هاذا": "هذا",
    "هده": "هذه",
    "هدي": "هذه",
    "علشان": "عشان",
    "عشان": "من اجل",
    "الدواء": "دواء",
    "الدوا": "دواء",
    "الدول": "دواء",
    "حبه": "حبة",
    "حبات": "حبات",
    "كبسوله": "كبسولة",
    "معلقة": "ملعقة",
    "معلقه": "ملعقة",
    "نقطه": "نقطة",
    "تحميله": "تحميلة",
    "لبوسه": "لبوس",
}

NUM_WORDS = {
    "واحد": "1", "واحدة": "1",
    "اثنين": "2", "اتنين": "2",
    "ثلاث": "3", "تلات": "3",
    "اربعة": "4", "أربعة": "4",
    "خمسة": "5", "ستة": "6", "سبعة": "7",
    "ثمانية": "8", "تسعة": "9", "عشرة": "10"
}

PHRASE_RULES = [
    (r"\bقبل\s+الاكل\b", "قبل الاكل"),
    (r"\bبعد\s+الاكل\b", "بعد الاكل"),
    (r"\b(مع|اثناء)\s+الاكل\b", "مع الاكل"),
    (r"\bعالريق\b", "على الريق"),
    (r"\bعلي\s+الريق\b", "على الريق"),
    (r"\bعلى\s+الريق\b", "على الريق"),
    (r"\bقبل\s+الفطور\b", "قبل الفطور"),
    (r"\bبعد\s+الفطور\b", "بعد الفطور"),
    (r"\bقبل\s+النوم\b", "قبل النوم"),

    (r"\bمرة\s+باليوم\b", "مرة يوميا"),
    (r"\bمرتين\s+باليوم\b", "مرتين يوميا"),
    (r"\b(ثلاث|تلات)\s+مرات\s+باليوم\b", "3 مرات يوميا"),
    (r"\bكل\s*4\s*ساع(?:ة|ات)\b", "كل 4 ساعات"),
    (r"\bكل\s*6\s*ساع(?:ة|ات)\b", "كل 6 ساعات"),
    (r"\bكل\s*8\s*ساع(?:ة|ات)\b", "كل 8 ساعات"),
    (r"\bكل\s*12\s*ساع(?:ة|ات)\b", "كل 12 ساعة"),
    (r"\b(وقت|عند)\s+الل\w+وم\b", "عند اللزوم"),

    (r"\b(ثلاث|تلات)\s+ايام\b", "3 ايام"),
    (r"\bعشرة\s+ايام\b", "10 ايام"),
    (r"\bاسبوعين\b", "2 اسبوع"),
    (r"\bشهرين\b", "2 شهر"),
]

def normalize_text_base(text: str) -> str:
    text = str(text or "").strip()

    text = AR_DIACRITICS.sub("", text)
    text = TATWEEL.sub("", text)

    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و").replace("ئ", "ي")

    text = text.translate(AR_INDIC_TO_LATIN).translate(FA_INDIC_TO_LATIN)

    text = re.sub(r"(\d),(\d)", r"\1.\2", text)

    text = PUNCT_SPACES.sub(r" \1 ", text)
    text = EXTRA_SPACES.sub(" ", text).strip()

    return text


def normalize_text_strong(text: str) -> str:
    text = normalize_text_base(text)

    keys = sorted(NUM_WORDS.keys(), key=len, reverse=True)
    pat = r"\b(" + "|".join(map(re.escape, keys)) + r")\b"
    text = re.sub(pat, lambda m: NUM_WORDS[m.group(0)], text)

    for pat, rep in PHRASE_RULES:
        text = re.sub(pat, rep, text)

    keys = sorted(WORD_MAP_STRONG.keys(), key=len, reverse=True)
    pat = r"\b(" + "|".join(map(re.escape, keys)) + r")\b"
    text = re.sub(pat, lambda m: WORD_MAP_STRONG[m.group(0)], text)

    text = EXTRA_SPACES.sub(" ", text).strip()
    return text


MODEL_NAME = "openai/whisper-large-v3"

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="arabic", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to("cuda", dtype=torch.float16)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")

print("Model loaded FP16 on GPU.")


def load_audio_16k(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = torch.tensor(audio, dtype=torch.float32)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    return audio


def apply_vad_chunks(waveform, sample_rate=16000, frame_ms=30, energy_threshold=0.0005):
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len <= 0:
        return [waveform.numpy()]

    chunks = []
    current = []

    for i in range(0, len(waveform), frame_len):
        frame = waveform[i:i+frame_len]
        if len(frame) < frame_len:
            break
        energy = (frame**2).mean().item()
        if energy > energy_threshold:
            current.append(frame)
        else:
            if current:
                chunks.append(torch.cat(current))
                current = []
    if current:
        chunks.append(torch.cat(current))

    if not chunks:
        return [waveform.numpy()]
    return [c.numpy() for c in chunks]


def transcribe_batch(paths, use_vad=True):
    all_audios = []

    for p in paths:
        wf = load_audio_16k(p)
        if use_vad:
            chunks = apply_vad_chunks(wf, sample_rate=16000)
            if len(chunks) > 1:
                sep = torch.zeros(int(0.2 * 16000))
                merged = []
                for c in chunks:
                    merged.append(torch.tensor(c))
                    merged.append(sep)
                wf_merged = torch.cat(merged)
                all_audios.append(wf_merged.numpy())
            else:
                all_audios.append(chunks[0])
        else:
            all_audios.append(wf.numpy())

    inputs = processor(all_audios, sampling_rate=16000, return_tensors="pt", padding=True)
    mel = inputs.input_features

    max_len = 3000
    T = mel.shape[-1]

    if T < max_len:
        pad_len = max_len - T
        mel = F.pad(mel, (0, pad_len), mode="constant", value=0.0)
    elif T > max_len:
        mel = mel[:, :, :max_len]

    mel = mel.to("cuda", dtype=torch.float16)

    with torch.no_grad():
        pred_ids = model.generate(
            mel,
            temperature=0,
            num_beams=5
        )

    texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return [normalize_text_strong(t) for t in texts]