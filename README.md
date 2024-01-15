# Reccommender Systems

This project was developed with Julia Moska and Jakub Polczyk

## Data

We used Amazon Review Data from https://nijianmo.github.io/amazon/index.html 

Data was cleaned. Minimum 10 reviews per item. Every user wrote min. 3 reviews. Data was splitted to train and test sets.

## Model

- Model is sequential according to below modalities:
- User reviews about purchased clothes were collected.
- Using a deep learning model, the sentiment of each statement was determined based on the text.
- The sentiment was mapped to numerical values.
- A table was generated, where user IDs were in rows, item IDs were in columns, and sentiment values were the entries.
- This table was used to determine the nearest neighbors for a given user.
- The collaborative part of the system returns a specified number of recommendations based on cosine similarity.
- Another component of the system is a deep neural network, whose task is to create a ranking among items proposed by the collaborative part.
- The input to the network is the user ID and an image of the item for which we want to know the probability that the user will like it.
- The training example looked like this: user_id, image -> sentiment (on a scale from 0 to 1).
- The network learned user embeddings.
- The network component was a ResNet without the classification layer.
- For each recommended item from the previous step, the probability that the user would like the item was obtained.
- Recommendations were sorted in descending order.
- The obtained ranking was the final ranking.