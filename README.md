# 🩺 Disease Prediction Project  
A simple GUI-based system that predicts diseases based on symptoms using Machine Learning. Built during an internship at Varcons Technologies Pvt Ltd.  

## 🧠 Features  
- Predicts one of 41 diseases based on user-input symptoms  
- GUI built using Tkinter  
- Uses Naive Bayes classifier (GaussianNB)  
- Trained on a cleaned dataset of 4920 records and 132 features  
- Final model saved as .pkl and used in live prediction  

## 🚀 How to Run the Project  
1. Clone the Repository  
git clone https://github.com/moulyamc/Disease_prediction_project.git  
cd Disease_prediction_project  

2. Install Dependencies  
pip install -r requirements.txt  

3. Train the Model (optional – only if gaussian_nb_model.pkl is missing)  
python src/train_model.py  

4. Launch the GUI  
python src/GUI.py  

## 🗂️ Project Structure  
Disease_prediction_project/  
├── data/  
│   └── cleaned_file.csv  
├── model/  
│   └── gaussian_nb_model.pkl  
├── src/  
│   ├── train_model.py  
│   └── GUI.py  
├── requirements.txt  
├── .gitignore  
├── LICENSE  
└── README.md  

## 🧪 ML Algorithms Used  
- Gaussian Naive Bayes  
- Logistic Regression (mentioned in report)  
- Random Forest (optional/compared)  

## 👩‍💻 Author  
Moulya M C  
B.E. in Computer Science, VTU  

## 📝 License  
This project is licensed under the MIT License.  
For academic and demonstration purposes.