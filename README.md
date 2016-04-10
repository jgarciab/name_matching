# LSH forest approach using 3-grams and tf-idf to match names to a database
Based on http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LSHForest.html

Input to train:
- List of all names in the database

Input to match:
- List of names to match

Limitations:
- Names are not associates to IDs now (easy to implement by saving another list/dict with the IDs)
- Names are only matched by cosine similarity (easy to implement by adapting filter_data_exact)
- Memory hungry! (Max. number of names in the database around 500k names / Gb RAM)
