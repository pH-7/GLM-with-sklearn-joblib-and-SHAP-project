import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shap
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def fit_glm(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def explain_model(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("outputs/shap_summary.png")
    plt.close()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig("outputs/shap_feature_importance.png")
    plt.close()

def generate_pdf_report(classification_rep_path="outputs/classification_report.txt"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("outputs/report.pdf", pagesize=A4)
    elements = []

    elements.append(Paragraph("GLM Model Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    with open(classification_rep_path, "r") as f:
        report_text = f.read()
    elements.append(Paragraph("<b>Classification Report:</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<pre>{report_text}</pre>", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>SHAP Summary Plot:</b>", styles["Heading2"]))
    elements.append(Image("outputs/shap_summary.png", width=400, height=300))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>SHAP Feature Importance:</b>", styles["Heading2"]))
    elements.append(Image("outputs/shap_feature_importance.png", width=400, height=300))

    doc.build(elements)

def main():
    df = load_data("data/sample.csv")
    feature_cols = ["age", "income", "balance"]
    X, y = preprocess_data(df, "default", feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = fit_glm(X_train, y_train)
    y_pred = model.predict(X_test)
    report_str = classification_report(y_test, y_pred)
    print(report_str)

    with open("outputs/classification_report.txt", "w") as f:
        f.write(report_str)

    joblib.dump(model, "models/glm_model.pkl")
    explain_model(model, X_train)
    generate_pdf_report()


# Ensure the outputs directory exists
# Ensure we are in the correct directory
if __name__ == "__main__":
    main()
