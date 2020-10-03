---
title: "Weather and Crime Analysis in Louisville, KY"
date: 2020-05-20
tags: [Python, weather, crime, data science]
header:
  image: "/images/louisville.jpg"
  caption: "Photo credit: alexeys/iStock.com"
excerpt: "This project analyzes the weather temperature and crime statistics in Louisville, KY"
---

For my full analysis of this project, please refer to [City of Louisville, KY Analysis](https://nbviewer.jupyter.org/github/thanhnguyenduong/DSC540_Weather_and_Crime_analysis_in_Louisville-KY/blob/master/Thanh%20Nguyen-Duong%20DSC540%20Milestone%205.ipynb)

For this project, I used 3 datasets that were obtained through three different methods:

CSV file: [Crime in Louisville Dataset 2003-2017](https://www.kaggle.com/jpayne/crime-in-louisville-ky-2003-2017)
Website: [Zip-codes.com](https://www.zip-codes.com/state/ky.asp)
API: [OpenWeatherMap.org](https://openweathermap.org/api)

### Objective
The idea is since I want to travel to Louisville, KY for a vacation during the summer. I wanted to see how crime rate in this city has changed over time. In addition, which Zip Code has the most crime counts; since Zip Code is the common variable because for the next two datasets I will use Zip Code to get weather information through OpenWeatherMap and to see the total demographics of that particular Zip Code. I will use this website data file and scrape Zip Codes for Kentucky and use those Zip Codes to pull weather data from the next source of data through the use of API key. Overall, the goal of the project is that I can get weather information for every zip code available in the city of Louisville, KY and also see how the crime rate through different types of crimes are.

### Accomplishments
This project gives me an opportunity to understand how to web scrape data from a website using BeautifulSoup, though Scrapy can also be used here. Aside from web scraping, I had to be able to request and get data through an API key and combined all of these data from different sources(csv, website and API) to create a master dataframe and analyze from there. 

### I hope you enjoy reading this analysis of mine through the link above!
