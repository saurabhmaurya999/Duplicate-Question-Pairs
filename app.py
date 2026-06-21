
from flask import Flask, render_template, request
import pickle
import numpy as np
from gensim.models import KeyedVectors

app = Flask(__name__)

try:
    model = pickle.load(open("model.pkl", "rb"))
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("MODEL ERROR:", e)


try:
    w2v_model = KeyedVectors.load("word2vec.kv")
    print("W2V LOADED SUCCESSFULLY")
except Exception as e:
    print("W2V ERROR:", e)


def avg_w2v(text):
    words = text.split()

    vectors = [
        w2v_model[word]
        for word in words
        if word in w2v_model
    ]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)


def basic_features(q1, q2):
    q1_words = q1.split()
    q2_words = q2.split()

    common_words = len(set(q1_words) & set(q2_words))
    total_words = len(set(q1_words)) + len(set(q2_words))
    word_share = common_words / total_words if total_words != 0 else 0

    return np.array([
        len(q1),
        len(q2),
        len(q1_words),
        len(q2_words),
        common_words,
        total_words,
        word_share,
        0,
        0
    ])


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        q1 = request.form["question1"]
        q2 = request.form["question2"]

        vec1 = avg_w2v(q1)
        vec2 = avg_w2v(q2)

        features = basic_features(q1, q2)
        final_input = np.hstack((features, vec1, vec2)).reshape(1, -1)

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            result = "Questions are Not Duplicate"
        else:
            result = "Both questions are Duplicate"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)