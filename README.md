# NYCAirbnbRegression
### Project Overview:
Cost is a big problem in many parts of the world, especially in Bay Area. As someone who first lived in New York city then San Francisco, the two most expensive cities in the U.S., I am always interested in understanding the attributes of living cost. Kaggle provides a great dataset of New York city Airbnb prices that allows me to explore what factors play a role in the cost of Airbnb.

This dataset includes customers’ reviews, neighborhood, borough, latitude and longitude, room type, and other information of each Airbnb unit instance. In this project, I will explore the following questions.

- How does the Airbnb price vary based on its location?
- Do different room types lead to different prices?
- What is the overall sentiment of the customers’ reviews?
- Can we predict Airbnb prices with a model?

### Data source: 
https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

### Future Studies:
So far, from the input features, including neighborhood groups, neighborhoods, availability in a year, last review, the best regression model is only moderate, with an R2 value around 0.5. 

One hypothesis I formed was that there were too many distracting features. I tested this out by only providing the neighborhood information with or without the room type information to the model, as they were shown to have impact on the prcies in the first two parts of the project. However, removing input features decreased the accuracy of the model. 

I took a step back and asked myself what do I care when I book an Airbnb. Information such as last review date, number of reviews, and room type is important. But for a visitor to New York, what matters most is how close the Airbnb unit is to the big apple's landmark and how safe it is. The next thing I will try is to import other databases to this dataset and determine how these new factors impact the prices. 

Stay tuned. 



