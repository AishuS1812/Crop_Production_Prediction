# 🌾 Crop Production Prediction

This project predicts **total crop production (in tons)** using agricultural data such as **area harvested, yield, and year**.  
It uses **Machine Learning models (Linear Regression, Random Forest, XGBoost)** and provides an interactive **Streamlit dashboard** for EDA, model evaluation, and predictions.

---

## 📂 Project Structure
```
crop-production-prediction/
├─ app.py                        # Streamlit dashboard
├─ requirements.txt              # Python dependencies
├─ README.md                     # Project documentation
├─ .gitignore                    # Git ignore rules
├─ LICENSE                       # Open-source license
├─ data/                         # Dataset folder (ignored by git)
│   └─ FAOSTAT_data.xlsx
├─ notebooks/                    # Jupyter notebooks
│   └─ Crop_Production.ipynb
├─ presentation/                 # Project presentation
│   └─ Crop_Production_Prediction_Presentation.pptx
└─ images/                       # Visuals for documentation
    ├─ eda_infographic.png
    └─ dashboard_preview.png
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/crop-production-prediction.git
cd crop-production-prediction
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Add dataset
- Place `FAOSTAT_data.xlsx` into the `data/` folder.  
- This file is **ignored by git** (see `.gitignore`) because of its size.

### 4. Run Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Features
- **EDA (Exploratory Data Analysis)**  
  - Crop distribution (Top 10 crops).  
  - Production trends over years.  
  - Correlation heatmaps & scatter plots.  

- **Modeling**  
  - Compare Linear Regression, Random Forest, XGBoost.  
  - Metrics: R², MAE, MSE.  
  - Feature importance visualization.  

- **Prediction**  
  - User selects Region, Crop, Year → App predicts **total production in tons**.  
  - Shows both predicted and actual values (if available).  
  - Option to provide manual inputs for Area harvested & Yield.  

---

## 📈 Example Use Cases
- **Governments & NGOs** → food security planning.  
- **Policymakers** → subsidies, crop insurance, relief programs.  
- **Agri-businesses** → optimize storage & transportation.  
- **Farmers & Traders** → plan sales with market forecasts.  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** – interactive dashboard  
- **Pandas, NumPy** – data processing  
- **Matplotlib, Seaborn** – visualization  
- **scikit-learn, XGBoost** – machine learning  

---

## 📦 Requirements
See `requirements.txt`

---

## 📜 License
This project is licensed under the **MIT License** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements
- Dataset: [FAOSTAT](https://www.fao.org/faostat/en/)  
- Streamlit for rapid web app development.  
