import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommender", layout="wide")

# LOAD DATA
df = pd.read_csv("songs.csv")

df = df[['song', 'artist', 'genre']]
df['tags'] = df['artist'] + " " + df['genre']

# ML MODEL
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

# RECOMMEND FUNCTION
def recommend(song):
    index = df[df['song'] == song].index[0]
    distances = similarity[index]
    songs = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    return [df.iloc[i[0]].song for i in songs]

# UI
st.markdown("<h1 style='text-align:center; color:#1DB954;'>🎵 Music Recommendation System</h1>", unsafe_allow_html=True)

selected_song = st.selectbox("🎶 Choose a song", df['song'].values)

if st.button("🚀 Recommend"):
    recs = recommend(selected_song)

    st.markdown("## 🎧 Recommended Songs")

    cols = st.columns(3)

    for i, song in enumerate(recs):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#121212;padding:15px;border-radius:10px;text-align:center;color:white">
                <img src="https://i.imgur.com/8zQZ6sG.png" width="100">
                <h4>{song}</h4>
                <p>🎤 {df[df['song']==song]['artist'].values[0]}</p>
            </div>
            """, unsafe_allow_html=True)
