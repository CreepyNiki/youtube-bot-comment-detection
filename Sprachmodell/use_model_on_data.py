import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from sklearn.metrics import classification_report

# Daten laden mit verschiedenen Datensatzgrößen
data = pd.read_csv("Sprachmodell/comments_with_label.csv")
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
    
    # Modellvorhersage generieren
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1,  # Nur 1 neues Token generieren
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,
        )
    
    # Den vollständigen Modelloutput dekodieren
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Nur den tatsächlichen Modelloutput nach dem Prompt extrahieren
    actual_answer = decoded_output[len(prompt):].strip()
    
    # Nur das erste Zeichen des Outputs berücksichtigen
    label = actual_answer[0] if actual_answer else "unknown"
    
    return int_to_label[int(label)] if label in ["0", "1"] else "unknown", decoded_output

# Vorhersagen generieren
predictions = []
decoded_outputs = []

# Generieren von Predictions
for comment in data['Comment']:
    prediction, decoded_output = classify_comment(comment)
    predictions.append(prediction)
    decoded_outputs.append(decoded_output)

data['Predictions'] = predictions
data['DecodedOutputs'] = decoded_outputs

# Wahre Labels in numerisches Format umwandeln
true_labels = data['Label'].map(label_to_int).tolist()

# Vorhersagen in numerisches Format umwandeln
predicted_labels = data['Predictions'].map(label_to_int).fillna(-1).astype(int).tolist()  # -1 für "unknown"

# Ergebnisse ausgeben
for comment, true_label, predicted_label, decoded_output in zip(data['Comment'], true_labels, predicted_labels, data['DecodedOutputs']):
    
    # Wahres Label und vorhergesagtes Label von allen Kommentaren ausgeben
    # print(f"True label: {int_to_label[true_label]}")
    # print(f"Predicted label: {int_to_label[predicted_label]}")
    
    # Modelloutput und Kommentar ausgeben
    # kommentar_pos = decoded_output.find("Kommentar:")
    # if kommentar_pos != -1:
    #     print(f"Model output: {decoded_output[kommentar_pos:]}")

    # Kommentar, wahres und predictetes Label ausgeben, wenn das Modell eine falsche Vorhersage getroffen hat
    if true_label != predicted_label:
        print(f"Comment: {comment}")
        print(f"True label: {int_to_label[true_label]}")
        print(f"Predicted label: {int_to_label[predicted_label] if predicted_label in int_to_label else 'UNKNOWN PREDICTION'}")

# Statistiken ausgeben (Anzahl der Kommentare, Anzahl der richtigen und falschen Vorhersagen)
print("Total comments:", len(data['Comment']))
print("Total right predictions:", len([1 for true, pred in zip(true_labels, predicted_labels) if true == pred]))
print("Total wrong predictions:", len([1 for true, pred in zip(true_labels, predicted_labels) if true != pred]))

# Classification report ausgeben
print(classification_report(true_labels, predicted_labels))

