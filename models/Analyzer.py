import os
import re
import string
import stanza
from num2words import num2words


class AdvancedTurkishAnalyzer:
    def __init__(self, emolex_path="Data/Turkish-NRC-EmoLex.txt"):
        # 1. Türkçe ünlü harfler (Hece hesabı için)
        self.vowels = set("aeıioöuüAEIİOÖUÜ")

        # 2. Stanza Türkçe modelini başlat
        print("Stanza NLP Türkçe modeli yükleniyor... (İlk seferde kısa bir indirme yapabilir)")
        stanza.download('tr', processors='tokenize,pos,lemma,depparse')
        self.nlp = stanza.Pipeline('tr', processors='tokenize,pos,lemma,depparse', verbose=False)

        # 3. Sözlük kümesini başlat ve dosyayı oku
        self.abstract_keywords = set()
        self._load_nrc_emolex(emolex_path)

    def _load_nrc_emolex(self, path):
        """Paylaşılan geniş tab formatındaki NRC EmoLex dosyasını hatasız ayrıştırır."""
        if not os.path.exists(path):
            print(f"Hata: {path} konumunda sözlük dosyası bulunamadı! Boş küme ile devam ediliyor.")
            return

        print(f"NRC EmoLex sözlüğü yükleniyor: {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

            for index, line in enumerate(lines):
                # Başlık satırını atla
                if index == 0 or not line.strip():
                    continue

                parts = line.strip().split("\t")
                # 12 sütunlu tam yapıyı kontrol et
                if len(parts) >= 12:
                    # Sütun endeksleri:
                    # 1: anger, 3: fear, 5: negative, 7: sadness, 11: Turkish Word
                    anger = parts[1]
                    fear = parts[3]
                    negative = parts[6]  # Paylaşılan düzende sırayla bakınca 6. indeks negative'dir
                    sadness = parts[7]
                    turkish_word = parts[11].strip().lower()

                    # Eğer bu kelime negatif duygulardan biriyle eşleşiyorsa (1 ise)
                    if '1' in [anger, fear, negative, sadness]:
                        # "terk etmek" gibi çoklu yapıları parçalayıp kök kelimeleri kümeye ekle
                        # Böylece Stanza sadece kökü ("terk") yakalasa bile sözlükle eşleşebilir
                        for word in turkish_word.split():
                            # Parantez veya ek karakterleri temizle
                            cleaned_word = word.strip(string.punctuation)
                            if cleaned_word:
                                self.abstract_keywords.add(cleaned_word)

        print(
            f"Sözlük başarıyla yüklendi. Toplam {len(self.abstract_keywords)} adet bilişsel yük oluşturabilecek kelime kökü kayıt edildi.")

    def _convert_numbers_to_words(self, text: str) -> str:
        """Metindeki sayıları (örn: 2026) Türkçe okunuşlarına çevirir."""

        def replace_match(match):
            num_str = match.group(0)
            try:
                return num2words(int(num_str), lang='tr')
            except ValueError:
                return num_str

        return re.sub(r'\d+', replace_match, text)

    def analyze(self, text: str) -> dict:
        if not text.strip():
            return {"error": "Metin boş olamaz."}

        # Ön işleme: Sayıları yazıya çevirerek heceleme hatasını engelle
        processed_text = self._convert_numbers_to_words(text)

        # Stanza NLP Analizi
        doc = self.nlp(processed_text)

        total_words = 0
        total_syllables = 0
        long_words_count = 0

        passive_verb_count = 0
        adjective_count = 0
        adverb_count = 0
        abstract_word_count = 0
        unique_words = set()

        total_sentences = max(len(doc.sentences), 1)

        for sentence in doc.sentences:
            for word in sentence.words:
                # Noktalama işaretlerini kelime olarak sayma
                if word.upos == 'PUNCT':
                    continue

                total_words += 1
                word_text = word.text.lower()
                unique_words.add(word_text)

                # Hece hesabı (Ünlü harf sayımı)
                syllables = sum(1 for char in word.text if char in self.vowels)
                total_syllables += syllables
                if syllables >= 4:
                    long_words_count += 1

                # Dil Bilgisi Analizi (POS Tagging)
                if word.upos == 'ADJ':
                    adjective_count += 1
                elif word.upos == 'ADV':
                    adverb_count += 1
                elif word.upos == 'VERB':
                    # Türkçe edilgenlik morfolojisi kontrolü
                    if word.feats and 'Voice=Pass' in word.feats:
                        passive_verb_count += 1

                # Soyut / Bilişsel Yükü Yüksek Kelime Kontrolü (Stanza Köken Analizi ile)
                if word.lemma and word.lemma.lower() in self.abstract_keywords:
                    abstract_word_count += 1

        total_words = max(total_words, 1)

        # Metriklerin Hesaplanması
        avg_sentence_length = total_words / total_sentences
        avg_word_length_syllables = total_syllables / total_words
        long_word_ratio = (long_words_count / total_words) * 100

        # Klasik Okunabilirlik Formülleri
        atesman_score = 199.67 - (40.12 * avg_word_length_syllables) - (2.8 * avg_sentence_length)
        cetinkaya_score = 118.60 - (2.59 * avg_sentence_length) - (0.83 * long_word_ratio)

        # İleri Seviye Veri Bilimi Metrikleri
        lexical_diversity = len(unique_words) / total_words
        passive_ratio = (passive_verb_count / total_words) * 100
        abstract_ratio = (abstract_word_count / total_words) * 100

        # Bilişsel Yük Sınır Tayini (DEHB ve Disleksi İçin)
        cognitive_load = "Normal"
        if avg_sentence_length > 15 or abstract_ratio > 18 or passive_ratio > 10:
            cognitive_load = "Yüksek"

        return {
            "classical_stats": {
                "total_sentences": total_sentences,
                "total_words": total_words,
                "avg_sentence_length_words": round(avg_sentence_length, 2),
                "avg_word_length_syllables": round(avg_word_length_syllables, 2)
            },
            "readability_scores": {
                "atesman": round(atesman_score, 2),
                "cetinkaya_uzun": round(cetinkaya_score, 2)
            },
            "cognitive_load_metrics": {
                "cognitive_load_level": cognitive_load,
                "abstract_word_percentage": round(abstract_ratio, 2),
                "passive_voice_percentage": round(passive_ratio, 2),
                "lexical_diversity_ttr": round(lexical_diversity, 2),
                "adjective_adverb_density": round(((adjective_count + adverb_count) / total_words) * 100, 2)
            }
        }


# --- Test Alanı ---
if __name__ == "__main__":
    # Eğer dosya yolun tam olarak Data/Turkish-NRC-EmoLex.txt ise doğrudan çalışır
    analyzer = AdvancedTurkishAnalyzer(emolex_path="Data/Turkish-NRC-EmoLex.txt")

    test_metni = (
        "Yapay zeka sistemleri 2026 yılında eğitim süreçlerini tamamen değiştirecektir. "
        "Ancak, bireylerin maruz kaldığı ağır edebi ifadeler ve terk edilmiş eski kavramlar "
        "zihinsel yorgunluk üreterek dikkat eksikliği süreçlerini tetiklemektedir."
    )

    sonuc = analyzer.analyze(test_metni)
    import json

    print("\n--- ANALİZ ÇIKTISI ---")
    print(json.dumps(sonuc, indent=4, ensure_ascii=False))
