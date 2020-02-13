# NYCAirbnbRegression
### Project Overview:
Cost is a big problem in many parts of the world, especially in Bay Area. As someone who first lived in New York city then San Francisco, the two most expensive cities in the U.S., I am always interested in understanding the attributes of living cost. Kaggle provides a great dataset of New York city Airbnb prices that allows me to explore what factors play a role in the cost of Airbnb.

### Part I (New_York_Airbnb.ipynb)
This dataset includes customers’ reviews, neighborhoods, boroughs, GPS coordinates, room typse, and other information of each Airbnb unit instance. In this part of the project, I will first explore the following questions based on this Kaggle dataset.

- How does the Airbnb price vary based on its location?
- Do different room types lead to different prices?
- What is the overall sentiment of the customers’ reviews?
- Can we predict Airbnb prices with a model?

#### Data source: 
NYC Airbnb Prices - Kaggle
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

#### Future Studies:
So far, from the input features, including neighborhood groups, neighborhoods, availability in a year, last review dates, the best regression model works only moderately, with an R2 value around 0.5. 

One hypothesis I formed was that there were too many distracting features. I tested this out by only providing the neighborhood information with or without the room type information to the model, as they were shown to have impact on the prcies in the first two sections. However, removing input features decreased the accuracy of the model (data not shown).

I took a step back and asked myself what I care when I book an Airbnb. Information such as last review date, number of reviews, and room type is important. But for a visitor to New York, what matters most is how close the Airbnb unit is to the big apple's landmark and how safe it is. Next, I will import other databases and determine how these new factors impact the prices. 

### Part II (New_York_Airbnb2.ipynb)
In this part of the project, I will import two additional datasets of New York subway stations and crime records from New York State webpages and examine the following three questions:

- How does the Airbnb price correlate with the apartment's proximity to subway stations?
- Is the Airbnb price negatively correlated with the crime rate in the neighbourhood?
- Is the model better at predicting Airbnb prices with additional subway and crime information?

#### Data source: 
NYC Transit Subway Entrance And Exit Data - NY State government
https://data.ny.gov/Transportation/NYC-Transit-Subway-Entrance-And-Exit-Data-API/rwat-jhj8
NYPD Arrests Data - Government Catalog
https://catalog.data.gov/dataset?tags=crime&organization=city-of-new-york

Work in progress. Stay tuned. 
