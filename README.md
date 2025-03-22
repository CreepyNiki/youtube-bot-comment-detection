# Youtube Bot Comment Detection

Ein universitäres Projekt im AM1 für das Seminar: "Anwendungen der Computerlinguistik". Dieses Projekt umfasst einen Datensatz annotierter deutschsprachiger YouTube Kommentare.

Ziel ist die automatische Klassifikation von Bot-Kommentaren durch zwei Ansätze.

1. Die automatische Klassifkation durch ein Feed-Forward-Neural-Network.
2. Die automatische Klassifikation als Ausgabe eines Sprachmodells durch Prompting.

## Setup

### Youtube-Comment-Extractor-Script
Um das Script für den CommentExtractor zu testen, ist Zugriff auf mein YouTube API Projekt vonnöten. Dafür können Sie mir gerne Ihre Gmail-Adresse zukommen lassen.

### Feature-Based-Machine Learning
1. Ohne Embeddings: Starten des Python Scripts ´neural_network.py´.
2. Mit Embeddings: Starten des Python Scripts ´neural_network_pre_trained_embeddings.py´.

### Large Language Model
1. Herunterladen und Entpacken des Scripts von Google Drive.
2. Einfügen des Ordners ´EleutherAI_gpt-neo-125M´ in die Dateistruktur.
3. Starten des Python Scripts ´use_model_on_data.py´.

Google Drive Link: [Sprachmodell](https://drive.google.com/drive/folders/1DhLnXPGenlY8Z28JD5GL4cf6kkfzPNDr?usp=drive_link)
