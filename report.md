# OCR assignment report

## Feature Extraction (Max 200 Words)
For my feature extraction I began by calculating the PCA’s from the training data, as this helped represent the multivariate data in a smaller number of dimensions. Only the training data was used for the PCA calculation, and the largest 40 (only 40 to ensure the model would not exceed the size limit) were stored in the model to later use during testing and evaluation stages. To calculate these, I found the covariance matrix and subsequently the eigenvectors of this. After, I decided to calculate a sum of 1-D divergences across all class comparisons to find what PCA features were most useful for classification. Although 1-D divergence usually isn’t effective, when coupled with the PCA features this obtained me more successful results. As PC1 accounts for a lot of variance in the data, it would come out as the most divergent feature, however PC1 does not separate the classes well. After calculating divergence, I selected the best features by penalising high correlations between PCA's. 

## Classifier (Max 200 Words)
To classify my data, I began by implementing a ‘Nearest Neighbour’ algorithm, using the cosine rule to find the closest data point from the training set to classify the test data, as this is a highly accurate algorithm. I was able to implement it as all the training data was supplied during the testing and evaluation stages. Working off this, I have created a ‘k Nearest Neighbour’ to find the average of the k nearest neighbours for the classification. For better results, I used a trial and error technique while changing the value of k, helping find the optimal value. Additionally, I have added a weight to the nearest neighbours, this will favour the closer neighbours to improve the classification. Similarly, to the k value, I went through multiple tests while changing the value of the weight to find an optimal value. I have found that using a larger k value is more effective when testing on noisier data. Unsurprisingly, changing the weight also had a similar effect, with smaller weights favouring noisier pages.

## Error Correction (Max 200 Words)
For my error correction I firstly imported a list of words (obtained from ‘http://www.mieliestronk.com/wordlist.html’) to store into the dictionary model. Noticing it was lacking any words with apostrophes, I appended these to the start of the file (sourced from: ‘https://www.panopy.com/iphone/secret-ada/cracking-a-cipher.html#apostrophewords’). During the testing phase, once the ‘correct_errors’ function was called, the word dictionary was taken from the model. Then, the labels were iterated through and the bounding boxes were checked for spaces by comparing the x coordinates between a single bounding box and the consecutive one. This then generated a list of potential words that were passed into another function. The words were then searched for within the dictionary, and if they were valid words they got returned. If the word was invalid, the algorithm then checks over any words in the dictionary that had the same length, scoring words based off their similarity (number of characters that were different). I felt anymore than two characters changed would be stretching so once a score of over two was reached, a score of -1 was returned, thus decreasing the search time. However, correction was ineffective when the classification scores were low, as more characters were incorrect.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: 94.0%
- Page 2: 93.9%
- Page 3: 85.6%
- Page 4: 66.4%
- Page 5: 47.5%
- Page 6: 38.2%

## Other information (Optional, Max 100 words)
I decided to implement a gaussian blur to the training data in order to improve the performance of my solution, particularly on more noisy pages. To do this I used ‘numpy.random.normal’ to generate the noise and added it to half of the training data, before feature selection. For the inputs, I used a negative mean of the training data, as well as experimenting with variation of the standard deviation input to achieve the best result. This drastically increased performance as it allowed the system to train against noise. 