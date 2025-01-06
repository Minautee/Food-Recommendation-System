## Recipe Recommendation System 🥘

### Overview
This project involves developing a recipe recommendation system using the Food.com Recipes and User Interactions dataset. The system provides personalized recipe recommendations by implementing content-based filtering, collaborative filtering, and hybrid recommendation techniques. The dataset contains recipes, user ratings, reviews, and associated metadata.

### Approach
#### 1. **Dataset Preparation**
The dataset from Food.com Recipes and User Interactions was used to build the foundation for this project. It contains detailed recipe information such as ingredients, tags, reviews, and user ratings. The dataset was pre-processed and enriched as follows:

-> **Difficulty Level:** Calculated based on the n_steps column, with thresholds defining "Easy," "Medium," and "Hard."
-> **Tags and Ingredients Conversion:** Transformed string representations of lists into proper Python lists for easier processing.
-> **Nutritional Tags:** Established thresholds for key nutritional values (calories, protein, fat, etc.) and assigned tags like "Low Fat" or "High Protein." Extracted nutritional tags from the existing tags list and appended relevant ones to each recipe.
-> **Review Cleaning:** Applied NLP techniques to clean and preprocess user reviews, removing noise and extracting meaningful words.
-> **Sentiment Analysis:** Performed sentiment analysis on reviews to determine positive or negative sentiments, aiding in identifying liked or disliked ingredients.
-> **Ingredient Sentiment Mapping:** Identified liked and disliked ingredients based on sentiment scores of user reviews.
-> **Tag Enrichment:** Extracted additional information such as meal type (e.g., breakfast, lunch), cuisine type, and preference (veg or non-veg) from the tags column.

#### 2. Recommendations
Three recommendation techniques were implemented:

-> **Content-Based Filtering:**
* Utilized cosine similarity on user preferences (e.g., ingredients, dietary restrictions, meal type) and recipe data (e.g., tags and ingredients) to rank recipes.
* TF-IDF vectorization was applied to encode ingredients and compute similarity.

-> **Collaborative Filtering:**
* Constructed a user-item matrix (pivot table of user ratings for recipes) based on the suggestions from content-base filtering.
* Applied Singular Value Decomposition (SVD) to predict missing ratings and rank recipes for users.

-> **Hybrid Recommendations:**
* Combined the results of content-based and collaborative filtering.
* Calculated a hybrid score using a weighted combination of content similarity and predicted ratings.

### Challenges Faced

1. **Resource Limitations:**

The dataset's large size required significant computational resources, especially for creating similarity matrices. To manage this, a subset of 30,000 rows was sampled for processing and recommendations.

2. **Single Dataset Limitation:**

Using only one dataset restricted the diversity and breadth of recommendations. Incorporating additional datasets would have required significant cleaning and imputation of missing values, increasing the risk of inconsistent data.

3. **Complexity of Data Extraction:**

Extracting meaningful information like cuisine, nutritional tags, and meal types from the dataset's tags was challenging due to common words being attached to varied adjectives.

### Ideas for Improvement

1. **Integration of GenAI:**

* Use Generative AI models to enhance extraction of tags such as cuisine, meal type, and preferences, ensuring higher accuracy and reducing manual effort.
* Employ Retrieval-Augmented Generation (RAG) with these datasets to improve data enrichment and recommendation quality.

2. **Enhanced Recipe Links:**

Include direct links to recipe pages in the recommendations, providing users with easy access to full instructions.

3. **Ingredient Processing:**

Perform more granular analysis of ingredients (e.g., grouping by food categories or substitutability) for more nuanced recommendations.

4. **Interactive User Feedback:**

Enable users to interact with recommendations by providing feedback, which can be used to fine-tune subsequent suggestions.

5. **Nutritional Insights:**

Provide detailed nutritional analysis for each recommendation, allowing users to make informed choices based on their dietary goals.
