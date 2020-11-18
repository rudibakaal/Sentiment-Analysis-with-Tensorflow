# Sentiment-Analysis-with-Tensorflow

## Motivation
A binary classification machine learning algorithm for determining Yelp review sentiments.

The dataset contains 1000 Yelp reviews labelled with positive or negative sentiments [1].

The algorithm was able to learn the message vocabulary and idf, then return the term-document matrix through the sklearn TfidfVectorizer class. 

## Neural Network Topology and Results Summary

The binary-crossentropy loss function was leveraged along with the Adam optimizer for this classification problem.

![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

After 35 epochs the binary and validation classifiers reaches 99% and 82% accuracy, respectively, in determining review sentiment. 

![yelp](https://user-images.githubusercontent.com/48378196/99531983-abbec800-29f7-11eb-8637-dcaa290263ac.png)

## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1] 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015
