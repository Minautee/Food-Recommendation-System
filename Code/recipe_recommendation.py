# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from wordcloud import WordCloud
import re, ast, sys
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import streamlit as st

# %%
raw_interactions_df = pd.read_csv(r"D:\Practice\Food Recommendation System\Food.com Recipes and Interactions\RAW_interactions.csv")
raw_interactions_df.head()

# %%
raw_recipes_df = pd.read_csv(r"D:\Practice\Food Recommendation System\Food.com Recipes and Interactions\RAW_recipes.csv")
raw_recipes_df.head()

# %%
raw_interactions_df.shape
raw_recipes_df.shape

# %%
combined_df = raw_recipes_df.merge(raw_interactions_df, left_on='id', right_on='recipe_id')
combined_df.head()

# %%
combined_df.drop(["submitted","contributor_id","id"],axis=1,inplace=True)

final_df = combined_df.reindex(columns=['recipe_id', 'name', 'date', 'user_id', 'minutes', 'tags', 'nutrition', 'n_steps', 'description', 'ingredients', 'n_ingredients', 'rating', 'review'])
final_df.head()

# %%
final_df = final_df.fillna('NA')

# %%
final_df.info()
final_df.isnull().sum()

# %%
subset_df = final_df.sample(30000)
subset_df.head()

# %%
def calculate_difficulty(n_steps):
    if n_steps <= 5:
        return 'Easy'
    elif 6 <= n_steps <= 10:
        return 'Medium'
    else:
        return 'Hard'

subset_df['difficulty'] = subset_df['n_steps'].apply(calculate_difficulty)

# %%
# 2. Parse ingredients and tags
def preprocess_column(value, split=False):
    if pd.isna(value):
        return []
    if split:
        return [item.strip() for item in value.split(',')]
    else:
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

subset_df['ingredients'] = subset_df['ingredients'].apply(lambda x: preprocess_column(x, split=True))
subset_df['tags'] = subset_df['tags'].apply(lambda x: preprocess_column(x))

# %%
# --- NLP for User Preferences ---
import string
def preprocess_text(text):

    #1 Remove Punctuation
    #2 Remove stopwords

    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    spamwords=[word for word in nopunc.split() if word.lower in stopwords.words('english')]

    return clean_words

subset_df['cleaned_review'] = subset_df['review'].apply(preprocess_text)

# %%
# 3. Infer dietary restrictions from nutrition
subset_df['nutrition'] = subset_df['nutrition'].apply(lambda x: eval(x) if isinstance(x, str) else x)
nutrition_columns = ['calories','total fat','sugar','sodium','protein','saturated fat','carbohydrates']
subset_df[nutrition_columns]= pd.DataFrame(subset_df['nutrition'].tolist(), index=subset_df.index)

# Verify the columns
print(subset_df[nutrition_columns].describe())

# %%
# Set up the plotting environment
plt.figure(figsize=(15, 10))
for i, col in enumerate(nutrition_columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(subset_df[col], bins=20, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Nutrition Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# %%
# Calculate quantile thresholds for each metric
thresholds = {
    'calories': subset_df['calories'].quantile(0.75),       # High calorie if above 75th percentile
    'protein': subset_df['protein'].quantile(0.75),         # High protein if above 75th percentile
    'carbohydrates': subset_df['carbohydrates'].quantile(0.25),  # Low carb if below 25th percentile
    'sugar': subset_df['sugar'].quantile(0.75),             # High sugar if above 75th percentile
    'saturated_fat': subset_df['saturated fat'].quantile(0.75),  # High saturated fat if above 75th percentile
    'total_fat': subset_df['total fat'].quantile(0.75),     # High fat if above 75th percentile
}

# Print thresholds for review
print("Thresholds for dietary tags:")
for key, value in thresholds.items():
    print(f"{key}: {value}")

# %%
# Function to assign dietary tags
def assign_dietary_tags(row):
    tags = []
    if row['calories'] > thresholds['calories']:
        tags.append('High Calorie')
    if row['protein'] > thresholds['protein']:
        tags.append('High Protein')
    if row['carbohydrates'] < thresholds['carbohydrates']:
        tags.append('Low Carb')
    if row['sugar'] > thresholds['sugar']:
        tags.append('High Sugar')
    if row['saturated fat'] > thresholds['saturated_fat']:
        tags.append('High Saturated Fat')
    if row['total fat'] > thresholds['total_fat']:
        tags.append('High Fat')
    return tags

# Apply function to the DataFrame
subset_df['dietary_tags'] = subset_df.apply(assign_dietary_tags, axis=1)

# Review the resulting dietary tags
print(subset_df[['name', 'dietary_tags']].head())

# %%
def extract_dietary_tags(row):
    try:
        # Convert tags from string to list if necessary
        tags = ast.literal_eval(row['tags']) if isinstance(row['tags'], str) else row['tags']
    except (ValueError, SyntaxError):
        tags = []  # Handle invalid or empty tags gracefully

    # Create regex patterns for keywords with adjectives like high/low
    patterns = [fr"(?:high|low|reduced|free)?\s*{keyword}" for keyword in nutrition_columns]
    
    # Find tags that match the patterns
    dietary_tags = [tag for tag in tags if any(re.search(pattern, tag, re.IGNORECASE) for pattern in patterns)]
    return dietary_tags

def update_dietary_tags(row):
    # Extract new dietary tags using the function
    new_tags = extract_dietary_tags(row)
    # Combine existing and new tags
    combined_tags = set(row['dietary_tags'] + new_tags)  # Use set to avoid duplicates
    return list(combined_tags)  # Convert back to list for consistency

# Apply the function to update dietary_tgs
subset_df['dietary_tags'] = subset_df.apply(update_dietary_tags, axis=1)
print(subset_df['dietary_tags'])

# %%
from collections import Counter

# Count occurrences of each dietary tag
all_tags = [tag for tags in subset_df['dietary_tags'] for tag in tags]
tag_counts = Counter(all_tags)

# Plot tag distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=list(tag_counts.keys()), y=list(tag_counts.values()), palette='Set2')
plt.title("Distribution of Dietary Tags")
plt.xlabel("Dietary Tags")
plt.ylabel("Number of Recipes")
plt.xticks(rotation=45)
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Dietary Tags Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# %%
subset_df['cleaned_review'] = subset_df['cleaned_review'].apply(lambda x: ', '.join(map(str, x)))
subset_df.head()

# %%
# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

subset_df['sentiment'] = subset_df['cleaned_review'].apply(analyze_sentiment)
subset_df['sentiment_label'] = subset_df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

# %%
subset_df.info()
subset_df.isnull().sum()

# %%
modified_df = subset_df.copy()
modified_df.head()

# %%
modified_df['ingredients'].iloc[0]

# %%
# Define a function to identify meal type based on reviews
def identify_meal_type(row):
    # print(row)
    tags = row['tags']
    # print(tags)
    # print(tags.type)
    review = row['cleaned_review']
    meal_types = ['breakfast', 'brunch', 'lunch', 'snacks', 'dinner', 'desserts', 'other']
    
    # Check for meal type keywords in tags or reviews
    for meal in meal_types:
        if any(meal in tag.lower() for tag in tags):  # Check each tag in the list
            return meal
    
    # Check for meal type keywords in review
    for meal in meal_types:
        if re.search(meal, review, re.IGNORECASE):  # Check for keywords in review
            return meal
    
    return 'other'

def identify_cuisine(row):
    # print(row)
    tags = row['tags']
    cuisine = ['indian', 'italian', 'chinese', 'continental', 'mexican', 'japanese', 'thai', 'mediterranean', 'other']
    
    # Check for meal type keywords in tags or reviews
    for meal in cuisine:
        if any(meal in tag.lower() for tag in tags):  # Check each tag in the list
            return meal
    
    return 'other'

# Define a function to determine preference (veg or non-veg)
def determine_preference(row):
    non_veg_items = [
        'chicken', 'beef', 'pork', 'fish', 'egg', 'eggs', 'lamb', 
        'bacon', 'sausage', 'shrimp', 'turkey', 'salmon', 'crab', 'mutton', 'lobster'
    ]
    veg_keywords = ['veg', 'vegetarian', 'vegan']
    
    # Convert tags and ingredients to lowercase for comparison
    ingredients_lower = [ingredient.lower() for ingredient in row['ingredients']]
    tags_lower = [tag.lower() for tag in row['tags']]
    name_lower = row['name'].lower()
    
    # Check recipe name for non-veg items
    if any(item in name_lower for item in non_veg_items):
        return 'non-veg'
    
    # Check tags for veg/vegetarian/vegan keywords
    if any(veg_keyword in tag for veg_keyword in veg_keywords for tag in tags_lower):
        return 'veg'
    
    # Check ingredients for non-veg items
    if any(item in ingredients_lower for item in non_veg_items):
        return 'non-veg'
    
    # Default to veg if no non-veg items are found
    return 'veg'

# Function to clean ingredient names
def clean_ingredient_name(ingredient):
    if pd.isna(ingredient):
        return None
    return ingredient.strip("[]'\"").lower()  # Remove brackets, quotes, and convert to lowercase

# Function to extract liked and disliked ingredients
def extract_preferences(row):
    liked = []
    disliked = []
    for ingredient in row['ingredients']:
        cleaned_ingredient = clean_ingredient_name(ingredient)
        if row['sentiment'] > 0:
            liked.append(cleaned_ingredient)
        elif row['sentiment'] < 0:
            disliked.append(cleaned_ingredient)
    return pd.Series([liked, disliked])


# Apply functions to extract preferences
modified_df[['liked_ingredients', 'disliked_ingredients']] = modified_df.apply(extract_preferences, axis=1)

# Add meal_type column
modified_df['cuisine'] = modified_df.apply(identify_cuisine, axis=1)

# Add meal_type column
modified_df['meal_type'] = modified_df.apply(identify_meal_type, axis=1)

# Add preference (veg/non-veg) column
modified_df['preference'] = modified_df.apply(determine_preference, axis=1)

# %%
modified_df.head()

# %%
# Plot recipe distribution by meal type
sns.countplot(x='meal_type', data=modified_df)
plt.title("Recipe Distribution by Meal Type")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Meal Type Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# Visualize average nutrition (calories) by meal type
modified_df.groupby('meal_type')['calories'].mean().plot(kind='bar', color='skyblue')
plt.title("Average Calories by Meal Type")
plt.ylabel("Calories")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Calories Vs Meal Type Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf() 

# Plot recipe distribution by cuisine
sns.countplot(x='cuisine', data=modified_df)
plt.title("Recipe Distribution by Cuisine")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Cuisine Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# Plot recipe distribution by preference
sns.countplot(x='preference', data=modified_df)
plt.title("Recipe Distribution by Preference")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Preference.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# %%
# Flatten the lists of liked and disliked ingredients
liked_exploded = modified_df.explode('liked_ingredients')['liked_ingredients'].dropna()
disliked_exploded = modified_df.explode('disliked_ingredients')['disliked_ingredients'].dropna()

# Count occurrences
liked_counts = liked_exploded.value_counts().reset_index()
liked_counts.columns = ['ingredient', 'liked_count']

disliked_counts = disliked_exploded.value_counts().reset_index()
disliked_counts.columns = ['ingredient', 'disliked_count']

# Merge liked and disliked counts
ingredient_counts = pd.merge(
    liked_counts, disliked_counts, on='ingredient', how='outer'
).fillna(0)

# %%
# Normalize counts for heatmap
ingredient_counts['liked_count_norm'] = ingredient_counts['liked_count'] / ingredient_counts['liked_count'].sum()
ingredient_counts['disliked_count_norm'] = ingredient_counts['disliked_count'] / ingredient_counts['disliked_count'].sum()

# Select top 20 and bottom 20 ingredients by liked and disliked counts
top_20 = ingredient_counts.nlargest(20, 'liked_count')
bottom_20 = ingredient_counts.nsmallest(20, 'liked_count')
top_bottom_20 = pd.concat([top_20, bottom_20])

# Plotting a bar plot for popularity (Top and Bottom 20)
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_bottom_20.melt(id_vars='ingredient', value_vars=['liked_count', 'disliked_count']),
    x='value', y='ingredient', hue='variable', palette='coolwarm'
)
plt.title("Distribution of Top and Bottom 20 Liked and Disliked Ingredients")
plt.xlabel("Count")
plt.ylabel("Ingredients")
plt.legend(title="Sentiment")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Top and Bottom 20 Ingredients.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# %%
# Add veg/non-veg classification to the ingredient counts
ingredient_counts['type'] = modified_df['preference']

# Aggregate liked and disliked counts by type (Veg/Non-Veg)
type_distribution = ingredient_counts.groupby('type')[['liked_count', 'disliked_count']].sum().reset_index()

# Plotting the distribution
plt.figure(figsize=(8, 6))
sns.barplot(
    data=type_distribution.melt(id_vars='type', value_vars=['liked_count', 'disliked_count']),
    x='type', y='value', hue='variable', palette='coolwarm'
)
plt.title("Distribution of Liked and Disliked Ingredients by Type (Veg/Non-Veg)")
plt.xlabel("Ingredient Type")
plt.ylabel("Count")
plt.legend(title="Sentiment")
if 'streamlit' in sys.modules:
    # In Streamlit, we don't show the plot
    # Optionally save the plot to a file
    plt.savefig("Veg or Non-veg Like Vs Dislike Distribution.png")  # Save the plot to a file if needed
else:
    # If not in Streamlit (e.g., in Jupyter Notebook), show the plot
    plt.show()

# Clear the current figure if needed
plt.clf()  

# %%
modified_df.columns

# %%
columns_to_drop = ['calories', 'total fat',
       'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates', 'sentiment', 'sentiment_label']
recommendation_df = modified_df.drop(columns=columns_to_drop)
recommendation_df = recommendation_df.reset_index(drop=True)

# %%
recommendation_df.head()
recommendation_df.shape

# %%
recommendation_df.to_csv("Sorted Recommendation.csv")

# %%
# --- Recommendation Functions ---
# Content-Based Filtering
def content_based_recommendation(preferences, df, n_recommendations=5):
    # Extract user preferences
    user_ingredients = preferences.get('ingredients', [])
    user_dietary_tags = preferences.get('dietary_restrictions', [])
    user_tags = preferences.get('tags', [])
    
    # Build TF-IDF matrix for ingredients
    vectorizer = TfidfVectorizer(stop_words='english')
    ingredients_tfidf = vectorizer.fit_transform(df['ingredients'].apply(lambda x: ' '.join(x)))
    
    # Calculate similarity with user preferences
    user_profile = ' '.join(user_ingredients)
    user_vector = vectorizer.transform([user_profile])
    similarity = cosine_similarity(user_vector, ingredients_tfidf).flatten()
    
    # Filter based on dietary tags and other constraints
    filtered_df = df[
        (df['dietary_tags'].apply(lambda x: any(tag in user_dietary_tags for tag in x))) &
        (df['tags'].apply(lambda x: any(tag in user_tags for tag in x))) 
    ]
    
    # Rank by similarity
    filtered_df['similarity'] = similarity[filtered_df.index]
    recommendations = filtered_df.sort_values('similarity', ascending=False).head(n_recommendations)
    
    return recommendations[['name', 'ingredients', 'minutes', 'tags', 'difficulty', 'similarity']]

# %%
# Collaborative Filtering
def collaborative_filtering(preferences, df, n_recommendations=5):
    # Filter recipes based on user constraints
    user_item_matrix = modified_df.pivot_table(index='user_id', columns='name', values='rating', fill_value=0)
    print(user_item_matrix.shape)
    
    if user_item_matrix.shape[1] < 2:
        return "Insufficient data for collaborative filtering."
    
    # Apply SVD
    svd = TruncatedSVD(n_components=min(50, user_item_matrix.shape[1] - 1), random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_
    predicted_ratings = np.dot(user_factors, item_factors)
    
    # Rank recommendations
    recommended_indices = np.argsort(predicted_ratings.flatten())[::-1][:n_recommendations]
    recommended_indices = recommended_indices[recommended_indices < user_item_matrix.shape[1]]
    recommended_titles = user_item_matrix.columns[recommended_indices]
    return df[df['name'].isin(recommended_titles)][['name', 'ingredients', 'minutes', 'tags', 'difficulty', 'rating']]


# %%
# Hybrid Recommendation
def hybrid_recommendation(preferences, df, alpha=0.5, n_recommendations=5):
    content_recommendations = content_based_recommendation(preferences, df, n_recommendations * 2)
    collaborative_recommendations = collaborative_filtering(preferences, df, n_recommendations * 2)
    
    # Combine recommendations
    combined_df = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates()
    combined_df['score'] = alpha * combined_df['similarity'] + (1 - alpha) * combined_df['rating']
    return combined_df.sort_values('score', ascending=False).head(n_recommendations)

# %%
import streamlit as st

st.title("Recipe Recommendation System")

preferences = {
    'dietary_restrictions': st.multiselect("Dietary Restrictions", ['High Protein', 'Low Carb', 'High Sodium', 'High Carb', 'High Calories']),
    'ingredients': st.text_input("Ingredients (comma-separated)").split(','),
    'tags': st.multiselect("Meal Type", ['breakfast', 'brunch', 'lunch', 'snacks', 'dinner', 'desserts', 'other']),
    'difficulty_level': st.selectbox("Difficulty Level", ['Easy', 'Medium', 'Hard']),
    'time_to_cook': st.slider("Max Time to Cook (minutes)", 1, 60, 30)
}

st.write("Recommendations:")
recommendations = hybrid_recommendation(preferences, recommendation_df)
st.write(recommendations)


# %%
import streamlit as st

st.title("User Preferences and Recipe Recommendations")

# User preferences from reviews
user_id = st.selectbox("Select User ID", user_preferences['user_id'])
preferences = user_preferences[user_preferences['user_id'] == user_id].iloc[0]

st.write("Liked Ingredients:", preferences['liked_ingredients'])
st.write("Disliked Ingredients:", preferences['disliked_ingredients'])

# Generate recommendations
recommendations = content_based_with_preferences(preferences, df)
st.write("Recommended Recipes:")
st.write(recommendations)



