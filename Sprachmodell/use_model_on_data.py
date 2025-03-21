import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import shap
import scikitplot as skplt
import numpy as np

def confusion_matrix():
    confusion_matrix = metrics.confusion_matrix(true_labels_filtered, predicted_labels_filtered)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=['bot', 'nonbot']).plot(cmap=plt.cm.Blues)
    plt.show()

def plot_charts():
    # Labels werden umbenannt, um sie in der Legende darzustellen
    print(true_labels_filtered)
    y_test_labels_renamed = ['Bot' if label == 0 else 'Nonbot' for label in true_labels_filtered]

    # Wahrscheinlichkeiten werden in ein Numpy-Array umgewandelt
    y_pred_proba_np = np.array(y_pred_proba)

    # Lift Curve wird geplottet
    skplt.metrics.plot_lift_curve(y_test_labels_renamed, y_pred_proba_np)
    plt.show()

    # Cumulative Gain Curve wird geplottet
    skplt.metrics.plot_cumulative_gain(y_test_labels_renamed, y_pred_proba_np)
    plt.show()
    


# Daten laden mit verschiedenen Datensatzgrößen
data = pd.read_csv("Extract_Comments/comments_with_label.csv")
# data = pd.concat([data.head(200), data.tail(200)])
# data = pd.concat([data.head(150), data.tail(150)])
# data = pd.concat([data.head(100), data.tail(100)])
# data = pd.concat([data.head(50), data.tail(50)])
data = pd.concat([data.head(25), data.tail(25)])
# print(data)

label_to_int = {'bot': 0, 'nonbot': 1}
int_to_label = {v: k for k, v in label_to_int.items()}

# Modell und Tokenizer laden
model = AutoModelForCausalLM.from_pretrained("EleutherAI_gpt-neo-125M/model_saved")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI_gpt-neo-125M/tokenizer_saved")

# Modell auf das richtige Gerät verschieben
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Funktion zur Generierung von Vorhersagen
def classify_comment(comment):
    # Prompt vorbereiten
    prompt = (
        f"Stell dir vor du bist CyberSecurity Experte bei Youtube.\n"
        f"Bitte klassifiziere den folgenden Kommentar als Bot (0) oder Nonbot (1). Antworte bitte nur der Zahl 0 oder 1.\n"
        f"Beispiel 1: NONBOTKOMMENTAR Antwort: 1\n"
        f"Beispiel 2: BOTKOMMENTAR Antwort: 0\n"
        f"Wenn du den Kommentar richtig klassifizierst bekommst du 1000 Euro Trinkgeld.\n"
        f"Kommentar: {comment}  Antwort:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    
    # Modellvorhersage generieren mit Logits
    with torch.no_grad():
        outputs = model(**inputs)  # Direkt die Logits aus dem Modell abrufen
    
    logits = outputs.logits[:, -1, :]  # Nehme die letzten Token-Logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Softmax für Wahrscheinlichkeiten

    # Dekodiere das generierte Token
    predicted_id = torch.argmax(logits, dim=-1)
    decoded_output = tokenizer.decode(predicted_id, skip_special_tokens=True).strip()

    # Extrahiere das vorhergesagte Label
    label = decoded_output[0] if decoded_output else "unknown"

    # Wahrscheinlichkeit für Bot (0) und Nonbot (1)
    prob_bot = probabilities[0, label_to_int['bot']].item()
    prob_nonbot = probabilities[0, label_to_int['nonbot']].item()
    prob = [prob_bot, prob_nonbot]  # Array für beide Klassen

    return int_to_label[int(label)] if label in ["0", "1"] else "unknown", prob, decoded_output

# Vorhersagen generieren
predictions = []
decoded_outputs = []
y_pred_proba = []


# Generieren von Predictions
for comment in data['Comment']:
    prediction, prob, decoded_output = classify_comment(comment)
    predictions.append(prediction)
    decoded_outputs.append(decoded_output)
    y_pred_proba.append(prob)

data['Predictions'] = predictions
data['DecodedOutputs'] = decoded_outputs

# Wahre Labels in numerisches Format umwandeln
true_labels = data['Label'].map(label_to_int).tolist()

# Vorhersagen in numerisches Format umwandeln
predicted_labels = data['Predictions'].map(label_to_int).fillna(-1).astype(int)
predicted_labels = [label for label in predicted_labels if label != -1]  # Entferne -1 für unbekannte Vorhersagen
true_labels = [label for label in true_labels if label != -1] 

# Ergebnisse ausgeben
for comment, true_label, predicted_label, decoded_output in zip(data['Comment'], true_labels, predicted_labels, data['DecodedOutputs']):
    
    # Wahres Label und vorhergesagtes Label von allen Kommentaren ausgeben
    print(f"True label: {int_to_label[true_label]}")
    print(f"Predicted label: {int_to_label[predicted_label]}")
    
    # Modelloutput und Kommentar ausgeben
    kommentar_pos = decoded_output.find("Kommentar:")
    if kommentar_pos != -1:
        print(f"Model output: {decoded_output[kommentar_pos:]}")

    # Kommentar, wahres und predictetes Label ausgeben, wenn das Modell eine falsche Vorhersage getroffen hat
    if true_label != predicted_label:
        print(f"Comment: {comment}")
        print(f"True label: {int_to_label[true_label]}")
        print(f"Predicted label: {int_to_label[predicted_label] if predicted_label in int_to_label else 'UNKNOWN PREDICTION'}")

# Statistiken ausgeben (Anzahl der Kommentare, Anzahl der richtigen und falschen Vorhersagen)
print("Total comments:", len(data['Comment']))
print("Total right predictions:", len([1 for true, pred in zip(true_labels, predicted_labels) if true == pred]))
print("Total wrong predictions:", len([1 for true, pred in zip(true_labels, predicted_labels) if true != pred]))

filtered_data = [(true, pred) for true, pred in zip(true_labels, predicted_labels) if pred != -1]
true_labels_filtered, predicted_labels_filtered = zip(*filtered_data) if filtered_data else ([], [])

# Classification report ausgeben
print(classification_report(true_labels_filtered, predicted_labels_filtered))

# confusion_matrix()
plot_charts()




