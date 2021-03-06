# mercari
Mercari Price Prediction Challenge
It is a project for price prediction of Mercari's online app which is a Japanese Ecommerce company founded in February 2013 by Japanese Serial Entrepreneur Shintaro Yamada. The data for this challenge can be found at https://www.kaggle.com/c/mercari-price-suggestion-challenge/data. 
There are three datasets given for this competition. They are :-
1)	train.tsv.7z – A compressed 7z tab separated file of size 74.3 MB which when uncompressed becomes close to 329 MB.
2)	test.tsv.7z – A compressed 7z tab separated file of size 33.97 MB which when uncompressed becomes close to 150 MB.
3)	test_stg2.tsv.zip – A compressed 7z tab separated file of size 294.37 MB which when uncompressed becomes close to 736 MB.
I am not able to upload intermediate files here because of the file size. There is a .hdf5 file for the best model weights which I am not able to upload because of the size of the file.
You can download the EDA, the first cut model solutions file which is Mercari_Price_Suggestion_Challenge_Models_with_DL.ipynb and the final model solution file which is Mercari_Price_Suggestion_Challenge_Prediction_Improved_Models_17022021.ipynb
The final.ipynb consists of two functions for predicting one unseen data point and another one for calculating the metrics.
The app.py file contains the code for taking the data from the html form and then predicting for the data point.
The templates.zip file contains the htl file index.html. Just download it and unzip it, save the file in a folder called template, run the app.py code and get the predictions. The saved model weights could not be uploaded because of the file size constraint.
