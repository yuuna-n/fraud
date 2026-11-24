# fraud
# Sistem de predicție a fraudelor și anomaliilor bancare

Proiect de an, UTM, Facultatea Calculatoare, Informatică și Microelectronică, gr. SI-251.

Sistemul calculează pentru fiecare tranzacție un scor de risc. Folosește patru metode, comparate între ele:

| Metodă | Tip | Ce întoarce |
|---|---|---|
| XGBoost | supravegheată | probabilitatea de fraudă |
| Isolation Forest | nesupravegheată | scor de anomalie (percentilă) |
| Local Outlier Factor | nesupravegheată | scor de anomalie (percentilă) |
| Reguli de referință | clasică | puncte pentru criteriile din tabelul 2.4 |

Aplicația are o interfață web (Streamlit) pentru verificarea unei tranzacții, analiza unui fișier, explorarea datelor, compararea modelelor, reglarea pragului și explicațiile SHAP.

---

## 1. Instalare

Este nevoie de Python 3.10 sau mai nou.

```bash
cd fraud_system
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Datele

Descărcați setul **Credit Card Transactions Fraud Detection Dataset** de pe Kaggle
(https://www.kaggle.com/datasets/kartik2112/fraud-detection). Puneți cele două fișiere în `data/raw/`:

```
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

**Fără setul Kaggle:** puteți genera date demonstrative cu aceeași structură. Sunt utile doar pentru a verifica funcționarea sistemului:

```bash
python tools/genereaza_date_demo.py --clienti 120
```

> Generatorul suprascrie fișierele din `data/raw/`. Interfața afișează o bandă galbenă cât timp modelele sunt antrenate pe date demonstrative.

## 3. Antrenarea

```bash
python train.py
```

Opțiuni utile:

| Opțiune | Efect |
|---|---|
| `--dezechilibru ponderi\|subesantionare\|smote` | metoda de tratare a dezechilibrului (tabelul 2.2); implicit `ponderi` |
| `--metoda-prag cost\|f1\|recall` | cum se alege pragul pe validare; implicit `cost` |
| `--optimizare` | caută hiperparametrii XGBoost pe validare (grila din `src/config.py`) |
| `--esantion-clienti 200` | folosește doar 200 de carduri, pentru o rulare rapidă |

Pe setul Kaggle complet (~1,85 mil. tranzacții) antrenarea durează câteva minute. Cel mai lent pas este LOF, care calculează distanțe între puncte. Rezultatele se salvează în `models/`:

- `sistem.joblib`: preprocesorul, modelele și pragurile;
- `rezultate.json`: metricile pe validare și testare, curbele, statisticile EDA, importanța SHAP;
- `scoruri_validare.parquet`: scorurile pe validare, pentru pagina *Pragul de decizie*;
- `istoric_referinta.parquet`: setul de testare curățat, folosit ca istoric al clienților.

## 4. Interfața

```bash
streamlit run app/app.py
```

Se deschide în browser la http://localhost:8501.

| Pagina | Ce face |
|---|---|
| Prezentare | starea sistemului și rezumatul rezultatelor |
| Verificare tranzacție | formular pentru o plată: verdict, scorul fiecărei metode, criteriile îndeplinite, explicația SHAP |
| Analiza unui fișier | încărcați un CSV; primiți scorurile, lista tranzacțiilor semnalate (descărcabilă) și metricile, dacă fișierul are `is_fraud` |
| Explorarea datelor | figurile 2.1–2.4, corelațiile, valorile extreme (IQR) |
| Compararea modelelor | precision, recall, F1, ROC-AUC, PR-AUC, curbele ROC și PR, matricele de confuzie, suma fraudelor oprite |
| Pragul de decizie | un cursor pentru prag; arată fraudele prinse, alarmele false și costul (pe validare) |
| Explicabilitate | importanța globală a caracteristicilor (media \|SHAP\|) |

## 5. Testele

```bash
python -m pytest tests -q
```

Testele verifică valorile caracteristicilor derivate pe un exemplu calculat de mână. Verifică și faptul că o tranzacție nu folosește informații din viitor.

---

## Structura codului

```
fraud_system/
├── train.py                  # antrenarea completă (cap. 2)
├── src/
│   ├── config.py             # toți parametrii: coloane, modele, praguri, reguli
│   ├── data_loading.py       # citire CSV, curățare, raport de calitate (§2.1, §2.3)
│   ├── features.py           # caracteristicile derivate, tabelul 2.3; criteriile, tabelul 2.4
│   ├── preprocessing.py      # valori lipsă, codificare, scalare – învățate doar pe antrenare (§2.3)
│   ├── split.py              # împărțirea cronologică (§2.5)
│   ├── imbalance.py          # ponderi / subeșantionare / SMOTE (tabelul 2.2)
│   ├── models/
│   │   ├── supervised.py     # XGBoost
│   │   ├── unsupervised.py   # Isolation Forest, LOF
│   │   └── rules.py          # regulile de referință
│   ├── threshold.py          # alegerea pragului (cost / F1 / recall)
│   ├── evaluate.py           # metrici și curbe
│   ├── explain.py            # valori SHAP (TreeSHAP nativ din XGBoost)
│   ├── eda.py                # statistici pentru analiza exploratorie (§2.2)
│   └── predict.py            # SistemFrauda – folosit de interfață
├── app/
│   ├── app.py                # punctul de intrare Streamlit și navigarea
│   ├── ui.py                 # stil, culori, formatare, încărcarea modelelor
│   └── views/                # câte un fișier pentru fiecare pagină
├── tools/genereaza_date_demo.py
├── tests/test_features.py
└── .streamlit/config.toml    # tema interfeței
```

## Decizii de implementare (de menționat în raport)

1. **Împărțirea cronologică.** Datele de antrenare sunt prima parte (în timp) din `fraudTrain.csv`. Datele de validare sunt ultimele 20%. Datele de testare sunt `fraudTest.csv`, folosit o singură dată.
2. **Istoricul se calculează separat pentru fiecare set** (§2.5). Așadar, la începutul validării și al testării, clienții nu au încă istoric. Pragul este ales pe validare, care are aceeași situație ca testarea, deci alegerea lui rămâne corectă.
3. **Fără scurgere de informație.** Fiecare caracteristică folosește numai tranzacțiile *anterioare* ale cardului. Medianele, codificările și scalarea se învață doar pe datele de antrenare. Codificarea prin rata țintei (`merchant`) folosește validarea încrucișată internă.
4. **Codificarea** (§2.3):
   - one-hot pentru `category` și `gender`;
   - codificare ordinală pentru `state`;
   - rata țintei pentru `merchant`;
   - frecvența pentru `job` și `city`.

   Vârsta (`age`) se calculează din `dob`. Coloanele cu date personale directe se exclud: nume, stradă, `trans_num`.
5. **Scorurile nesupravegheate** sunt transformate în percentile față de datele de antrenare, ca să poată fi comparate cu probabilitatea XGBoost. LOF se antrenează pe un eșantion de 40 000 de tranzacții, cu `n_neighbors = 100`. Valoarea a fost aleasă pe validare, dintre 20, 35 și 100.
6. **Pragul** minimizează costul total pe validare. O fraudă ratată costă implicit cât 10 alarme false (`COST_FRAUDA_RATATA` în `src/config.py`).
7. **SHAP.** Valorile se obțin direct din XGBoost (`pred_contribs=True`, algoritmul TreeSHAP). Nu este nevoie de pachetul `shap`.
