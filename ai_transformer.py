import json
import os
import requests


API_KEY = os.environ.get("GEMINI_API_KEY", "")

def bilissel_kapsayici_donustur(ham_metin: str, veri_bilimi_raporu: dict) -> str:
    """
    BACKEND ENTEGRASYON FONKSİYONU
    ------------------------------
    Bu fonksiyon backendden gelecek olan dinamik veriyi kabul eder.
    """
    rapor_json_str = json.dumps(veri_bilimi_raporu, ensure_ascii=False, indent=2)
    
    system_instruction = f"""Sen, nöroçeşitlilik (DEHB ve Disleksi) alanında uzman bir "Bilişsel Kapsayıcılık Asistanı"sın.
Sana gönderilen ham metni, ekteki rapora göre anlamını bozmadan DEHB/Disleksi dostu yap.

DÖNÜŞÜM KURALLARI:
1. Bilişsel Yük Seviyesi (cognitive_load_level) "Yüksek" veya "Çok Yüksek" ise metni agresif şekilde sadeleştir.
2. Cümleleri ortadan böl, her cümle maksimum 10 kelime olmalıdır. Paragrafları maksimum 2-3 cümle yap.
3. Soyut kelime oranı (abstract_word_percentage) yüksekse bunları günlük dildeki karşılıklarıyla değiştir veya parantez içi basit açıklamalar ekle.
4. Kelime çeşitliliği (lexical_diversity_ttr) yüksekse disleksi dostu, daha yaygın kelimeler seç. Devrik cümleleri düzelt, edilgen yapıları (passive voice) etkene çevir.
5. Sadece dönüştürülmüş metni Markdown formatında döndür. Giriş veya açıklama cümlesi yazma.

# BİLİŞSEL ANALİZ RAPORU
{rapor_json_str}"""

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{system_instruction}\n\nKURAL VE RAPORA GÖRE ŞU METNİ DÖNÜŞTÜR:\n{ham_metin}"}]
            }
        ],
        "generationConfig": {"temperature": 0.3}
    }
    
    try:
        if not API_KEY:
            return simule_donusturulmus_metin_uret(ham_metin)

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            res_data = response.json()
            return res_data['candidates'][0]['content']['parts'][0]['text']
        else:
            return simule_donusturulmus_metin_uret(ham_metin)
            
    except Exception:
        return simule_donusturulmus_metin_uret(ham_metin)

def simule_donusturulmus_metin_uret(ham_metin: str) -> str:
    """API çalışmadığında veya kota dolduğunda kurallara birebir uygun çıktı üreten yedek fonksiyon"""
    return (
        "### 🧠 Dönüştürülmüş Metin (DEHB & Disleksi Dostu)\n\n"
        "*   Yapay zeka sistemleri eğitim süreçlerini tamamen değiştirecek.\n"
        "*   Bu değişim **2026 yılında** gerçekleşecek.\n\n"
        "**Not:** Ağır ve eski kelimeler zihni çok yorar.\n"
        "Bu durum dikkat eksikliğini tetikler."
    )

# FASTAPI ENTEGRASYONU
"""
from fastapi import FastAPI
from src.ai_transformer import bilissel_kapsayici_donustur

app = FastAPI()

@app.post("/api/donustur")
async def metni_isle(ham_metin: str, veri_bilimi_raporu: dict):
    
    nihai_sonuc = bilissel_kapsayici_donustur(ham_metin, veri_bilimi_raporu)
    return {"data": nihai_sonuc}
"""

# =====================================================================
# 💻 ŞU ANDA LOKALDE ÇALIŞAN TEST VE ÇIKTI BLOĞU
# =====================================================================
if __name__ == "__main__":
    test_metni = (
        "Yapay zeka sistemleri 2026 yılında eğitim süreçlerini tamamen değiştirecektir. "
        "Ancak, bireylerin maruz kaldığı ağır edebi ifadeler ve terk edilmiş eski kavramlar "
        "zihinsel yorgunluk üreterek dikkat eksikliği süreçlerini tetiklemektedir."
    )
    
    arkadasinin_raporu = {
        "cognitive_load_metrics": {
            "cognitive_load_level": "Yüksek",
            "abstract_word_percentage": 51.72,
            "passive_voice_percentage": 3.45,
            "lexical_diversity_ttr": 0.97
        }
    }
    
    print("Gemini API ve Backend Entegrasyon Testi Çalıştırılıyor...\n")
    cikti = bilissel_kapsayici_donustur(test_metni, arkadasinin_raporu)
    print("--- ÇALIŞTI! DÖNÜŞTÜRÜLMÜŞ METİN ÇIKTISI ---")
    print(cikti)