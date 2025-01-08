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

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
TOKEN_FILE = "Extract_Comments/token.json"
CLIENT_SECRETS_FILE = "Extract_Comments/client_secret_916248569311-lc6st7pc4ib5r7jv8ogas342r076dl9s.apps.googleusercontent.com.json"

def get_authenticated_service():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        try:
            credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except google.auth.exceptions.RefreshError:
            os.remove(TOKEN_FILE)
            credentials = None

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            credentials = flow.run_local_server(port=8080)

            with open(TOKEN_FILE, "w") as token:
                token.write(credentials.to_json())

    return googleapiclient.discovery.build(
        "youtube", "v3", credentials=credentials
    )

def get_video_ids(youtube, videos=500, max_iterations=3):
    video_ids = set()
    next_page_token = None
    iteration_count = 0

    while len(video_ids) < videos:
        search_response = youtube.search().list(
            part="snippet",
            q="",
            type="video",
            regionCode="DE",
            relevanceLanguage="de",
            maxResults=50,
            order="relevance",
            pageToken=next_page_token,
            publishedAfter="2025-01-05T00:00:00Z",
            fields="items(id/videoId),nextPageToken",
        )

        try:
            search_response = search_response.execute()
        except googleapiclient.errors.HttpError as e:
            if e.resp.status == 403 and "quotaExceeded" in str(e.content):
                print("Quota exceeded. Stopping data collection.")
                break

        for item in search_response["items"]:
            try:
                video_id = item["id"]["videoId"]
                video_ids.add(video_id)
            except KeyError:
                continue

        next_page_token = search_response.get("nextPageToken")
        if not next_page_token or iteration_count >= max_iterations:
            break

        iteration_count += 1
        print(f"Collected {len(video_ids)} videos so far...")

    if os.path.exists("Extract_Comments/video_ids.json"):
        with open("Extract_Comments/video_ids.json", "r") as file:
            existing_video_ids = set(json.load(file))
            video_ids = video_ids.union(existing_video_ids)

        if len(video_ids) > 1000:
            print("The JSON file already contains over 1000 video IDs. New video IDs will not be added.")
            return
    else:
        with open("Extract_Comments/video_ids.json", "w") as file:
            json.dump(list(video_ids), file)

    with open("Extract_Comments/video_ids.json", "w") as file:
        json.dump(list(video_ids), file)

def get_comments(youtube, comments_per_video=2, search_terms=None):
    with open("Extract_Comments/video_ids.json", "r") as file:
        video_ids = json.load(file)

    unique_comments = set()
    all_comments = []
    for video_id in video_ids:
        print(f"Processing video ID: {video_id}")
        video_comments = []
        for keyword in search_terms:
            next_page_token = None
            while len(video_comments) < comments_per_video:
                print(f"Searching for keyword: {keyword} in video ID: {video_id}")
                try:
                    request = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        textFormat="plainText",
                        maxResults=min(100, comments_per_video - len(video_comments)),
                        pageToken=next_page_token,
                        fields="items(snippet/topLevelComment/snippet(authorDisplayName,textDisplay,videoId,channelId,publishedAt)),nextPageToken",
                        searchTerms=keyword
                    )
                    response = request.execute()
                except googleapiclient.errors.HttpError as e:
                    if e.resp.status == 403 and "quotaExceeded" in str(e.content):
                        print("Quota exceeded. Stopping data collection.")
                        return all_comments
                    elif e.resp.status == 403 and "disabled comments" in str(e.content):
                        print(f"Comments are disabled for video ID: {video_id}")
                        break
                    elif e.resp.status == 400 and "The API server failed to successfully process the request" in str(e.content):
                        print(f"Invalid request for video ID: {video_id} with keyword: {keyword}. Skipping this request.")
                        break
                    elif e.resp.status == 404 and "videoId" in str(e.content):
                        print(f"Video ID: {video_id} not found. Skipping this video.")
                        break
                    else:
                        raise

                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    # Generate a unique key for each comment
                    comment_key = (
                        snippet["authorDisplayName"].strip(),
                        snippet["textDisplay"].strip(),
                        snippet["videoId"]
                    )

                    if keyword.lower() in snippet["textDisplay"].lower() and comment_key not in unique_comments:
                        video_comments.append(snippet)
                        unique_comments.add(comment_key)

                    if len(video_comments) >= comments_per_video:
                        break

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

        all_comments.extend(video_comments)
        print(f"Collected {len(video_comments)} comments for video ID: {video_id}")

    return all_comments

def write_comments_to_csv(all_comments):
    existing_comments = set()

    # Read existing comments from the CSV file if it exists
    if os.path.exists('Extract_Comments/comments.csv'):
        with open('Extract_Comments/comments.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                existing_comments.add(tuple(row))

    # Add new comments to the set of existing comments
    for comment in all_comments:
            print(comment)
            author = comment["authorDisplayName"]
            text = comment["textDisplay"]
            video_id = comment["videoId"]
            channel_id = comment["channelId"]
            published_at = comment["publishedAt"]
            existing_comments.add((author, text, video_id, channel_id, published_at))

    # Write the combined set of comments back to the CSV file
    with open('Extract_Comments/comments.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Comment', 'Video_ID', 'Channel_ID', 'Published_At'])
        for comment in existing_comments:
            writer.writerow(comment)

def getpublishedTime(youtube, video_ids):
    published_times = {}
    for video_id in video_ids:
        request = youtube.videos().list(
            part="snippet",
            id=video_id,
            fields="items(snippet(publishedAt))"
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            print(f"Warning: No items found for video ID {video_id}")
            continue
        published_times[video_id] = items[0]["snippet"]["publishedAt"]
    return published_times

def extract_video_ids_from_file(filepath):
    video_ids = []
    with open(filepath, "r", encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            video_ids.append(row[2])  # Assuming the video ID is in the third column
    return video_ids

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
youtube = get_authenticated_service()

# get_video_ids(youtube, videos=1000, max_iterations=2)

search_terms = ["liebe", "arbeit", "dank", "video", "katze", "kanal", "idee"]
comments = get_comments(youtube, comments_per_video=20, search_terms=search_terms)
write_comments_to_csv(comments)





# video_ids_bot = extract_video_ids_from_file("Extract_Comments/fixed_bot.csv")
# video_ids_nonbot = extract_video_ids_from_file("Extract_Comments/fixed_nonbot.csv")
# video_ids = video_ids_bot + video_ids_nonbot

# published_times = getpublishedTime(youtube, video_ids)

# with open ("Extract_Comments/published_times.json", "w") as file:
#     json.dump(published_times, file)





