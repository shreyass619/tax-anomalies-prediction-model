import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QTextEdit, QTableWidget, QTableWidgetItem
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class TaxEvasionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tax Evasion Detection (Random Forest)")
        self.setGeometry(100, 100, 800, 800)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.button = QPushButton("Train Model")
        self.button.clicked.connect(self.train_model)
        self.layout.addWidget(self.button)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        self.layout.addWidget(QLabel("Classification Report"))
        self.layout.addWidget(self.report_box)

        self.table = QTableWidget()
        self.layout.addWidget(QLabel("Confusion Matrix"))
        self.layout.addWidget(self.table)

        self.metrics_table = QTableWidget()
        self.metrics_table.setRowCount(7)
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Description"])
        self.layout.addWidget(QLabel("Explanation of Metrics"))
        self.layout.addWidget(self.metrics_table)

        self.populate_explanations()

        self.layout.addWidget(QLabel("Enter User Data for Prediction"))

        self.inputs = {}
        for feature in ["Age", "Annual_Income", "Tax_Paid", "Deductions", "Loan_Amount"]:
            hbox = QHBoxLayout()
            label = QLabel(feature)
            line_edit = QLineEdit()
            self.inputs[feature] = line_edit
            hbox.addWidget(label)
            hbox.addWidget(line_edit)
            self.layout.addLayout(hbox)

        self.predict_button = QPushButton("Predict Fraud")
        self.predict_button.clicked.connect(self.predict_user)
        self.layout.addWidget(self.predict_button)

        self.prediction_result = QLabel("")
        self.layout.addWidget(self.prediction_result)

        self.model = None
        self.scaler = None

    def train_model(self):
        df = pd.read_csv("tax_evasion_india_dataset.csv")
        X = df.drop(columns=["Fraudulent"])
        y = df["Fraudulent"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        self.report_box.setPlainText(report)

        self.table.setRowCount(2)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Predicted 0", "Predicted 1"])
        self.table.setVerticalHeaderLabels(["Actual 0", "Actual 1"])
        for i in range(2):
            for j in range(2):
                self.table.setItem(i, j, QTableWidgetItem(str(cm[i][j])))

    def populate_explanations(self):
        explanations = [
            ("Precision", "Out of predicted positives, how many were correct."),
            ("Recall", "Out of actual positives, how many were detected."),
            ("F1-Score", "Harmonic mean of precision and recall."),
            ("Support", "Number of true samples per class."),
            ("Accuracy", "Overall correctness of the model."),
            ("Class 0", "Non-fraudulent taxpayers."),
            ("Class 1", "Fraudulent taxpayers.")
        ]
        for row, (metric, desc) in enumerate(explanations):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(desc))

    def predict_user(self):
        if self.model is None or self.scaler is None:
            self.prediction_result.setText("Please train the model first.")
            return

        try:
            input_data = [float(self.inputs[feature].text()) for feature in self.inputs]
            input_scaled = self.scaler.transform([input_data])
            pred = self.model.predict(input_scaled)[0]
            self.prediction_result.setText(f"Prediction: {'Fraudulent' if pred == 1 else 'Not Fraudulent'}")
        except Exception as e:
            self.prediction_result.setText(f"Error in prediction: {str(e)}")

app = QApplication(sys.argv)
window = TaxEvasionApp()
window.show()
sys.exit(app.exec_())
