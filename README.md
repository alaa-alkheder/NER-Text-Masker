# NER Text Masker

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-NER-FFD21E)
![GLiNER](https://img.shields.io/badge/GLiNER-zero--shot-6E40C9)
![Arabic](https://img.shields.io/badge/Arabic-%D8%B9%D8%B1%D8%A8%D9%8A-0A7E3E)
![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.2.0-blue)
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

A tool that hides personal data in Arabic and English **Word (.docx)** and **text (.txt)** files, and returns the file in the same format and layout.

```
1 File & labels  →  2 Masking method  →  3 Review  →  4 Download the masked file
```

## At a glance

| | |
|---|---|
| **Files** | `.docx` and `.txt` — Arabic, English, or mixed |
| **What it hides** | Names, organizations, locations, nationalities, dates, occupations, e-mails, phones, URLs, IP addresses, credit cards, IBANs, ID numbers, and any custom label you type |
| **Masking methods** | `tag`, `pseudonym`, `char_mask`, `partial`, `hash`, `fake`, `remove` |
| **Models** | `dslim/bert-base-NER` (pre-trained), `numind/NuNER_Zero` (zero-shot), `urchade/gliner_multi-v2.1` (Arabic), plus regex rules |
| **Interfaces** | Streamlit app, command line, Python function |
| **Version** | 1.2.0 · 56 tests |

## How to run

Requires Python 3.10 or newer.

```bash
cd ner-text-masker
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_models.py    # optional: download the models once (~3.5 GB)

streamlit run app.py                 # app: http://localhost:8501
```

Command line:

```bash
python cli.py --list
python cli.py samples/sample_ar.docx -l PERSON ORGANIZATION LOCATION PHONE EMAIL -m pseudonym
```

Tests:

```bash
python -m pytest -q
```

## Model tests on Colab

| Notebook | What it tests | Link |
|---|---|---|
| Model comparison | dslim vs NuNER Zero on a Tesla T4: accuracy, masking recall, threshold, CPU/GPU speed, long documents, Arabic | [Open in Colab](https://colab.research.google.com/drive/1ZUgg2Uo_3wP4LU4zcXNTUqq6w45ucRnd) |
| Evaluation on public datasets | 2000 sentences from the GMB dataset on Kaggle, and the ANERCorp test split with its two corrected versions from GitHub | [`notebooks/NER evaluation on Kaggle datasets.ipynb`](notebooks/NER%20evaluation%20on%20Kaggle%20datasets.ipynb) |

Both notebooks are in the [`notebooks/`](notebooks/) folder. To open one in Colab: *File → Upload notebook*, then *Runtime → Change runtime type → T4 GPU*.

## Documentation

Full documentation (in Arabic) is in [`report.md`](report.md): supported labels, masking methods, which model detects what, processing rules and settings, performance, known limitations, troubleshooting, and code structure.

## Author

**Alaa Alkheder**  
Master's student, Damascus University  
LinkedIn: [linkedin.com/in/alaa-alkheder](https://www.linkedin.com/in/alaa-alkheder/)



## Topics

`ner` · `named-entity-recognition` · `pii` · `pii-detection` · `data-anonymization` · `text-anonymization` · `privacy` · `gdpr` · `arabic-nlp` · `nlp` · `transformers` · `huggingface` · `gliner` · `zero-shot` · `bert` · `streamlit` · `docx` · `python`
