# 🇺🇿 Aspect Term Extraction with BiLSTM+CRF for Uzbek

Ushbu loyiha **Uzbek tilida yozilgan sharhlar**dagi **aspekt (mavzu) so‘zlarini** aniqlash uchun BiLSTM + CRF modelidan foydalanadi. Model Gensim yordamida yuklangan `.bin` formatdagi **word2vec** embeddingdan foydalanadi.

---

## 📦 Loyiha tarkibi

Ajoyib, endi sizga `README.md` faylini tayyorlab beraman — barcha bosqichlar, tavsiyalar va `.bin` fayl manbasi bilan to‘liq hujjatli ko‘rinishda. Quyidagi faylni loyihangizning **asosiy papkasiga** joylashtiring.

---

## 📄 `README.md`

```markdown
# 🇺🇿 Aspect Term Extraction with BiLSTM+CRF for Uzbek

Ushbu loyiha **Uzbek tilida yozilgan sharhlar**dagi **aspekt (mavzu) so‘zlarini** aniqlash uchun BiLSTM + CRF modelidan foydalanadi. Model Gensim yordamida yuklangan `.bin` formatdagi **word2vec** embeddingdan foydalanadi.

---

## 📦 Loyiha tarkibi

```

.
├── model/
│   └── bilstm\_crf\_model.py
├── utils/
│   ├── dataset.py
│   └── xml\_to\_bio.py
├── data/
│   └── train\_bio.json
├── embedding/
│   └── uzbek-word2vec.bin  ← 📥 Word2Vec fayl
├── train.py
├── test.py
├── inference.py
└── README.md

````

---

## 🔧 Talablar (Dependencies)

Python 3.8+ talab qilinadi.

```bash
pip install -r requirements.txt
````

`requirements.txt` tarkibi:

```text
torch
torchcrf
gensim
scikit-learn
seqeval
```

---

## 📥 Word2Vec Embedding (.bin)

Sizga kerakli `.bin` faylni quyidagi manzildan yuklab oling:

📎 [uzbek-word2vec.bin](https://github.com/RJalol/uzbek-sentiment-analysis-GCNN/blob/main/embedding/uzbek-word2vec.bin)

Uni shu joyga saqlang: `embedding/uzbek-word2vec.bin`

---

## 🧾 Ma’lumotlarni tayyorlash

Yoki siz `train_bio.json` faylini o’zingiz yaratishingiz mumkin BIO formatda:

```json
[
  {
    "tokens": ["Ovqat", "sifati", "yaxshi", "emas"],
    "labels": ["B-ASP", "I-ASP", "O", "O"]
  }
]
```

Agar siz HuggingFace'dagi [Sanatbek/aspect-based-sentiment-analysis-uzbek](https://huggingface.co/datasets/Sanatbek/aspect-based-sentiment-analysis-uzbek) datasetdan foydalanmoqchi bo‘lsangiz, `xml_to_bio.py` yordamida uni BIO formatga o‘tkazishingiz mumkin.

---

## 🚀 Treningni boshlash

```bash
python train.py
```

Model:

* `embedding/uzbek-word2vec.bin` dan embedding yuklaydi
* Trening yakunida `model/bilstm_crf.pth` faylini saqlaydi

---

## 🧪 Test qilish

```bash
python test.py
```

Natijalar quyidagi ko‘rinishda bo‘ladi:

```text
📊 Validation Result:
              precision    recall  f1-score   
         ASP       0.86      0.83      0.89      
```

---

## 🔍 Inference (bashorat qilish)

```bash
python inference.py
```

So‘ngra sizdan matn kiritish so‘raladi:

```text
✏️ Matn kiriting (so‘zlar bo‘shliq bilan ajratilgan):
Ovqat sifati yomon edi xizmat yaxshi emas
```

Natija:

```text
📌 Bashorat:
Ovqat           → B-ASP
sifati          → I-ASP
yomon           → O
edi             → O
xizmat          → B-ASP
yaxshi          → O
emas            → O
```

---

## ⚙️ Asosiy parametrlar (`train.py` dan)

```python
BATCH_SIZE = 16
EPOCHS = 40
LR = 0.0003
EMBEDDING_BIN_PATH = "embedding/uzbek-word2vec.bin"
HIDDEN_DIM = 256
MAX_LEN = 81
```

---

## 🧠 Model tuzilmasi

* **Embedding**: Word2Vec (`.bin`) dan yuklanadi
* **BiLSTM**: 2 yo‘nalishda, yashirin o‘lchami `256`
* **CRF**: Asosiy chiqish qatlam

---

## ✍️ Muallif

* Jaloliddin Rajabov ([@RJalol](https://github.com/RJalol))
* Aspect-Based Sentiment Analysis for Uzbek 🇺🇿

---

## 📜 Litsenziya

Ushbu loyiha o‘quv maqsadlarida foydalanish uchun ochiq.

```

---


```
