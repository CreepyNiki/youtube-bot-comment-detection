import csv
import os
import json
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_FILE = "token.json"
CLIENT_SECRETS_FILE = "client_secret_916248569311-clmn7dhj9sl47i15aso2h7igl1oeft9c.apps.googleusercontent.com.json"

def get_authenticated_service():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except google.auth.exceptions.RefreshError:
            # If the token has been expired or revoked, delete the token file
            os.remove(TOKEN_FILE)
            credentials = None

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            credentials = flow.run_console()

            with open(TOKEN_FILE, "w") as token:
                token.write(credentials.to_json())

    return googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials
    )

def getcomments():
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = get_authenticated_service()

    video_id = "QqD9X_p_L5E"
    all_comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        )

        response = request.execute()
        all_comments.extend(response.get("items", []))

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    # Print the total number of comments
    print(f"Total number of comments: {len(all_comments)}")

    # Print the comments with correct encoding for umlauts
    # print(json.dumps(all_comments, indent=4, ensure_ascii=False))

    # Write comments to CSV file
    with open('comments.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Comment', 'Video ID'])
        for comment in all_comments:
            snippet = comment["snippet"]["topLevelComment"]["snippet"]
            author = snippet["authorDisplayName"]
            text = snippet["textDisplay"]
            video_id = snippet["videoId"]
            writer.writerow([author, text, video_id])

def test():
    with open('comments.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if any("liebe" in cell.lower() for cell in row):
                print(row)            

    


# test()
getcomments()