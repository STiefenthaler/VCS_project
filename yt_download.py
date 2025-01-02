import yt_dlp

# Define the YouTube video URL
url = "https://www.youtube.com/shorts/6agKAltIlgw"

# Set download options
ydl_opts = {
    'format': 'best',
    'outtmpl': 'video.mp4',  # Save as 'video.mp4' in the current directory
}

# Download the video
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
