from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Dummy data (you can replace this with a CSV or DB later)
data = [
    {
        "descriptionOfTheProcurement": "Construction of Nyegezi Standa first phase",
        "entitySubCategoryName": "Medium and Large Works",
        "procurementCategoryName": "Works",
        "entityType": "PLANNED_TENDER",
        "uuid": "5127df8f-ea48-4f86-a35d-34a58575f549"
    },
    {
        "descriptionOfTheProcurement": "Supply of Office Furniture",
        "entitySubCategoryName": "Goods Supply",
        "procurementCategoryName": "Goods",
        "entityType": "ONGOING_TENDER",
        "uuid": "12345678-abcd-4e86-bbbb-34a58575f000"
    },
    {
        "descriptionOfTheProcurement": "Consultancy for Road Design",
        "entitySubCategoryName": "Consultancy Services",
        "procurementCategoryName": "Services",
        "entityType": "PLANNED_TENDER",
        "uuid": "abcdef12-3456-4e86-aacc-34a58575f111"
    }
]

df = pd.DataFrame(data)

# Combine features into one string for vectorization
def combine_features(row):
    return " ".join([
        row["descriptionOfTheProcurement"] or "",
        row["entitySubCategoryName"] or "",
        row["procurementCategoryName"] or "",
        row["entityType"] or ""
    ])

df["combined_features"] = df.apply(combine_features, axis=1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json
    if not user_data:
        return jsonify({"error": "No input provided"}), 400

    user_uuids = {item['uuid'] for item in user_data}
    user_indices = df[df['uuid'].isin(user_uuids)].index.tolist()

    if not user_indices:
        return jsonify({"error": "User UUIDs not found in data"}), 404

    # Average similarity across user items
    similarity_scores = cosine_sim[user_indices].mean(axis=0)

    # Exclude already selected items
    df["score"] = similarity_scores
    recommendations = df[~df["uuid"].isin(user_uuids)].sort_values(by="score", ascending=False)

    return jsonify(recommendations[["uuid", "descriptionOfTheProcurement", "score"]].to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)
