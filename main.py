from flask import Flask, redirect, render_template
import csv
import numpy as np

app = Flask(__name__)

books = []
liked = set()
liked_indexes = []

# Load topic vectors (e.g. LDA or Gibbs Sampling output)
topic_vectors = np.load('sol.npy')  # Shape: (num_books, 10)

# Load books.csv (Title, Author, Genre,...)
with open('books.csv', 'r', encoding='utf-8', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        books.append(row)

@app.route("/")
def index():
    return render_template("index.html", books=books, liked=liked)

@app.route("/like/<title>")
def like(title):
    if title not in liked:
        liked.add(title)
        for idx, book in enumerate(books):
            if book[0] == title:
                liked_indexes.append(idx)
                break
    return redirect("/")

@app.route("/reset")
def reset():
    liked.clear()
    liked_indexes.clear()
    return redirect("/")

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

@app.route("/recommended")
def recommended():
    if not liked_indexes:
        return render_template("recommended.html", recommendations=[])

    liked_vectors = topic_vectors[liked_indexes]
    avg_vector = liked_vectors.mean(axis=0)

    scores = []
    for idx in range(len(books)):
        if idx in liked_indexes:
            continue
        sim = cosine_similarity(avg_vector, topic_vectors[idx])
        scores.append((idx, sim))

    # Sort by similarity and take top 10
    top_recommendations = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    result = [(books[i][0], books[i][1], books[i][2], round(score, 3)) for i, score in top_recommendations]

    return render_template("recommend.html", recommendations=result)

if __name__ == "__main__":
    app.run(debug=True, port=3000)
