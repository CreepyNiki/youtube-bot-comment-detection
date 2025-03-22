import os
import json
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import random
from langdetect import detect, LangDetectException
import csv

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Bereich innerhalb der YouTube API, der genutzt wird
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
# Speicherort an dem das OAuth2.0 Token gespeichert wird
TOKEN_FILE = "Extract_Comments/token.json"
# Pfad zur Datei, die die Client ID und das Client Secret enthält
CLIENT_SECRETS_FILE = "Extract_Comments/client_secret_916248569311-lc6st7pc4ib5r7jv8ogas342r076dl9s.apps.googleusercontent.com.json"

# Funktion, die die Authentifizierung des Nutzers übernimmt
# Quelle: https://developers.google.com/people/quickstart/python?hl=de
def get_authenticated_service():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        # Falls das Token bereits existiert, wird es geladen
        try:
            credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except google.auth.exceptions.RefreshError:
            os.remove(TOKEN_FILE)
            credentials = None
           # Falls das Token nicht mehr gültig ist, wird es gelöscht und der Nutzer muss sich erneut authentifizieren
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            # Nutzer wird über den Browser authentifiziert
            credentials = flow.run_local_server(port=8080)

        # Das Token wird als token.json gespeichert
            with open(TOKEN_FILE, "w") as token:
                token.write(credentials.to_json())

    # Die YouTube API wird initialisiert und zurückgegeben
    return googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials
    )

# Funktion, die VideoIDs von YouTube sammelt und in einer JSON-Datei speichert
def get_video_ids(youtube, max_videos=500, max_iterations=3):
    # keine doppelten VideoIDs sollen vorhanden sein
    video_ids = set()
    next_page_token = None
    iteration_count = 0

    # Solange die Anzahl der gesammelten VideoIDs kleiner als die gewünschte Anzahl ist, werden weitere VideoIDs gesammelt
    while len(video_ids) < max_videos:
        # Suchparameter
        search_response = youtube.search().list(
            part="snippet",
            q="",
            type="video",
            regionCode="DE",
            relevanceLanguage="de",
            # pro Suchdurchgang können von der API aus nur maximal 50 Videos zurückgegeben werden -> Also wird die Anzahl der Suchdurchgänge erhöht (max_iterations)
            maxResults=50,
            order="relevance",
            pageToken=next_page_token,
            # Videos, die vor einem bestimmten Zeitpunkt veröffentlicht wurden, werden nicht berücksichtigt
            publishedAfter="2025-03-22T00:00:00Z",
            # Next Page token ermöglicht die mehreren Iterationen
            fields="items(id/videoId),nextPageToken",
        )


        # Error falls das Quota pro Tag der Youtube API aufgebraucht ist -> kostenlose Version hat ein Limit von 10000
        try:
            search_response = search_response.execute()
        except googleapiclient.errors.HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e.content):
                print("Quota exceeded. Stopping data collection.")
                break
        
        # Die VideoIDs werden in ein Set gespeichert, um doppelte VideoIDs zu vermeiden
        for item in search_response["items"]:
            try:
                video_id = item["id"]["videoId"]
                video_ids.add(video_id)
            except KeyError:
                continue
        
        # Next Page Token wird gesetzt, um die nächste Seite zu laden -> wenn keine Seite mehr vorhanden ist oder die maximalen Iterationen erfolgt sind, wird die Schleife beendet
        next_page_token = search_response.get("nextPageToken")
        if not next_page_token or iteration_count >= max_iterations:
            break
        
        # Anzahl der Iterationen wird erhöht und der Fortschritt wird nach jeder Iteration ausgegeben
        iteration_count += 1
        print(f"Collected {len(video_ids)} videos so far...")

    # Die gesammelten VideoIDs werden in eine JSON-Datei gespeichert
    if os.path.exists("Extract_Comments/video_ids.json"):
        with open("Extract_Comments/video_ids.json", "r") as file:
            existing_video_ids = set(json.load(file))
            # VideosIDs werden fusioniert, um doppelte VideoIDs zu vermeiden
            video_ids = video_ids.union(existing_video_ids)

        # maximale Länge des Files wird definiert
        if len(video_ids) > max_videos:
            print("The JSON file already contains over 1000 video IDs. New video IDs will not be added.")
            return
    else:
        # VideoIDs werden in die JSON-Datei geschrieben
        with open("Extract_Comments/video_ids.json", "w") as file:
            json.dump(list(video_ids), file)

    with open("Extract_Comments/video_ids.json", "w") as file:
        json.dump(list(video_ids), file)

# Funktion, die die Kommentare zu den gesammelten VideoIDs sammelt und in einer CSV-Datei speichert
def get_comments(youtube, comments_per_video=2, search_terms=None):

    # Datei mit den gesammelten VideoIDs wird geöffnet und geladen
    with open("Extract_Comments/video_ids.json", "r") as file:
        video_ids = json.load(file)

    # Set, um doppelte Kommentare zu vermeiden
    unique_comments = set()
    all_comments = []
    # Schleife über alle VideoIDs
    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")
        video_comments = []
        # Schleife über alle Suchbegriffe -> wird als Methodenparameter übergeben
        for keyword in search_terms:
            next_page_token = None
            # Es wird so lange nach Kommentaren gesucht, bis die gewünschte Anzahl erreicht ist
            while len(video_comments) < comments_per_video:
                print(f"Searching for keyword: {keyword} in video ID: {video_id}")
                try:
                    # Parameter für die Kommentarsuche
                    request = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        # pro Suchdurchgang können von der API aus nur maximal 100 Kommentare zurückgegeben werden. Also ist dies das Maximum oder die Anzahl der Kommentare, die noch fehlen.
                        maxResults=min(100, comments_per_video - len(video_comments)),                        
                        pageToken=next_page_token,
                        fields="items(snippet/topLevelComment/snippet(authorDisplayName,textDisplay,videoId,channelId,publishedAt)),nextPageToken",
                        searchTerms=keyword
                    )
                    # Request wird ausgeführt
                    response = request.execute()
                    # Error falls das Quota pro Tag der Youtube API aufgebraucht ist -> kostenlose Version hat ein Limit von 10000
                except googleapiclient.errors.HttpError as e:
                    if e.resp.status == 403 and "quotaExceeded" in str(e.content):
                        print("Quota exceeded. Stopping data collection.")
                        return all_comments
                        # Print falls die Kommentare eines Videos deaktiviert sind
                    elif e.resp.status == 403 and "disabled comments" in str(e.content):
                        print(f"Comments are disabled for video ID: {video_id}")
                        break
                     # Print falls es Probleme mit der API gibt
                    elif e.resp.status == 400 and "The API server failed to successfully process the request" in str(e.content):
                        print(f"Invalid request for video ID: {video_id} with keyword: {keyword}. Skipping this request.")
                        break
                    # Print falls ein Video nicht gefunden wurde -> auf privat gestellt oder gelöscht
                    elif e.resp.status == 404 and "videoId" in str(e.content):
                        print(f"Video ID: {video_id} not found. Skipping this video.")
                        break
                    else:
                        raise
                
                # Kommentare werden in ein Set gespeichert, um doppelte Kommentare zu vermeiden
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    # Generierung eines eindeutigen Schlüssels für jeden Kommentar -> betstehend aus Autor, Kommentar und VideoID
                    comment_key = (
                        snippet["authorDisplayName"].strip(),
                        snippet["textDisplay"].strip(),
                        snippet["videoId"]
                    )

                    # Kommentare werden nur hinzugefügt, wenn noch nicht in der Liste vorhanden sind
                    if keyword.lower() in snippet["textDisplay"].lower() and comment_key not in unique_comments:

                        video_comments.append(snippet)
                        unique_comments.add(comment_key)

                    if len(video_comments) >= comments_per_video:
                        break

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
        # Kommentare werden an die bestehende Liste aus der CSV Datei angehängt                        
        all_comments.extend(video_comments)
        print(f"Collected {len(video_comments)} comments for video ID: {video_id}")

    return all_comments

# Funktion, die die Kommentare in eine CSV-Datei schreibt
def write_comments_to_csv(all_comments):
    # Bereits vorhandene Kommentare werden in ein Set gespeichert, um doppelte Kommentare zu vermeiden
    existing_comments = set()

    # Bestehende Kommentare werden aus dem CSV-File gelesen und in das Set gespeichert
    if os.path.exists('Extract_Comments/comments.csv'):
        with open('Extract_Comments/comments.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                # Kommentare werden als Tupel gespeichert um die Daten nicht zu verändern
                existing_comments.add(tuple(row))

    # Neue Kommentare werden in das Set gespeichert und verschiedene Features werden extrahiert
    for comment in all_comments:
            print(comment)
            author = comment["authorDisplayName"]
            text = comment["textDisplay"]
            video_id = comment["videoId"]
            channel_id = comment["channelId"]
            published_at = comment["publishedAt"]
            existing_comments.add((author, text, video_id, channel_id, published_at))

    # Das kombinierte Set wird in die CSV-Datei geschrieben
    with open('Extract_Comments/comments.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Comment', 'Video_ID', 'Channel_ID', 'Published_At'])
        for comment in existing_comments:
            writer.writerow(comment)

# Funktion, die die VideoIDs aus einer CSV-Datei extrahiert -> als Grundlage für die Suche nach den Veröffentlichungszeiten
def extract_video_ids_from_file(filepath):
    video_ids = []

    # VideoIDs werden aus der CSV-Datei extrahiert
    with open(filepath, "r", encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            # Die VideoIDs werden in einer Liste gespeichert -> 3. Spalte der CSV-Datei
            video_ids.append(row[2])
    return video_ids

# Funktion, die die Veröffentlichungszeit der Videos sammelt und in einer JSON-Datei speichert
def getpublishedTime(youtube, video_ids):
    published_times = {}
    # Iteration über alle VideoIDs
    for video_id in video_ids:
        request = youtube.videos().list(
            part="snippet",
            id=video_id,
            fields="items(snippet(publishedAt))"
        )

        # Request wird ausgeführt
        response = request.execute()
        items = response.get("items", [])
        if not items:
            print(f"Warning: No items found for video ID {video_id}")
            continue
        # Veröffentlichungszeit wird in ein Dictionary gespeichert
        published_times[video_id] = items[0]["snippet"]["publishedAt"]
    return published_times


# Aufsetzung der Youtube API
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
youtube = get_authenticated_service()

get_video_ids(youtube, max_videos=1000, max_iterations=2)

# Suchbegriffe für die Kommentarsuche
search_terms = ["liebe", "arbeit", "dank", "video", "katze", "kanal", "idee"]
# # Kommentare werden gesammelt 
comments = get_comments(youtube, comments_per_video=1, search_terms=search_terms)
# # Kommentare werden in eine CSV-Datei geschrieben
write_comments_to_csv(comments)


# published_times = getpublishedTime(youtube, video_ids)
# with open ("Extract_Comments/published_times.json", "w") as file:
#     json.dump(published_times, file)





