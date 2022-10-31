---
title: Java - DukeJava 4-2 Step Two Simple recommendations
date: 2020-09-14 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

[toc]

---

# DukeJava 4-2 Step Two Simple recommendations

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 5.Java-Programming-Build-a-Recommendation-System
  - Step One : get the rating, rater, movie from the file
  - Step Two : Simple recommendations


Resource Link: https://www.coursera.org/learn/java-programming-recommender/supplement/KTrOQ/programming-exercise-step-two

starter files: https://www.dukelearntoprogram.com//course5/files.php

---

## file

```java
// ratedmovies_short.csv
id,title,year,country,genre,director,minutes,poster
0006414,"Behind the Screen",1916,"USA","Short, Comedy, Romance","Charles Chaplin",30,"http://ia.media-imdb.com/images/M/MV5BMTkyNDYyNTczNF5BMl5BanBnXkFtZTgwMDU2MzAwMzE@._V1_SX300.jpg"
0068646,"The Godfather",1972,"USA","Crime, Drama","Francis Ford Coppola",175,"http://ia.media-imdb.com/images/M/MV5BMjEyMjcyNDI4MF5BMl5BanBnXkFtZTcwMDA5Mzg3OA@@._V1_SX300.jpg"
0113277,"Heat",1995,"USA","Action, Crime, Drama","Michael Mann",170,"http://ia.media-imdb.com/images/M/MV5BMTM1NDc4ODkxNV5BMl5BanBnXkFtZTcwNTI4ODE3MQ@@._V1_SX300.jpg"
1798709,"Her",2013,"USA","Drama, Romance, Sci-Fi","Spike Jonze",126,"http://ia.media-imdb.com/images/M/MV5BMjA1Nzk0OTM2OF5BMl5BanBnXkFtZTgwNjU2NjEwMDE@._V1_SX300.jpg"
0790636,"Dallas Buyers Club",2013,"USA","Biography, Drama","Jean-Marc Vallée",117,"N/A"


// ratings_short.csv
rater_id,movie_id,rating,time
1,0068646,10,1381620027
1,0113277,10,1379466669
2,1798709,10,1389948338
2,0790636,7,1389963947
2,0068646,9,1382460093
3,1798709,9,1388641438
4,0068646,8,1362440416
4,1798709,6,1398043318
5,0068646,9,1364834910
5,1798709,8,1404338202



Movie (String anID, String aTitle, String aYear, String theGenres, String aDirector, String aCountry, String aPoster, int theMinutes)

Rater(String id, ArrayList<Rating>())

Rating (String anItem, double aValue)
```






---


## code

modify class SecondRatings:
- two private variables
- myMovies of type `ArrayList of type Movie`
  - `public Movie (String anID, String aTitle, String aYear, String theGenres)`
- named myRaters of `type ArrayList of type Rater`
  - `Rater(String id)`
- A default constructor
- Until you create the second constructor (see below), the class will not compile.


```java
import java.security.Principal;
import java.util.*;
import edu.duke.*;
import org.apache.commons.csv.*;

public class SecondRatings {

    private ArrayList<Movie> myMovies;
    private ArrayList<Rater> myRaters;

    public SecondRatings() {
        // default constructor
        // this("data/ratedmoviesfull.csv", "data/ratings.csv");
        this("ratedmovies_short.csv", "ratings_short.csv");
    }


// 1. Write an additional SecondRatings constructor
// has two String parameters named moviefile and ratingsfile.
// The constructor should create a FirstRatings object and then call the loadMovies and loadRaters methods in FirstRatings, to read in all the movie and ratings data, and store them in the two private ArrayList variables of the SecondRatings class, myMovies and myRaters.

    // loadMovies: -> ArrayList<Movie>
    // movieData [Movie, Movie, Movie, Movie, ...]
                    // id,title,year,country,genre,director,minutes,poster
                    // 0006414,"Screen",1916,"USA","Short, Comedy",30,"http://...jpg"

    // loadRaters: -> ArrayList<Rater>
    // -> raterData [Rater, Rater, Rater, ...]
    // -> raterList [String: [Rating, Rating, Rating], String: [Rating, Rating, Rating], ...]
                    // rater_id,movie_id,rating,time
                    // 1,0068646,10,1381620027

    public SecondRatings(String moviefile, String ratingsfile) {
        FirstRatings fratings = new FirstRatings();
        myMovies = fratings.loadMovies(moviefile);
        for(Movie m : myMovies){
            System.out.println(m);
        }
        myRaters = fratings.loadRaters(ratingsfile);
        for(Rater r : myRaters){
            System.out.println(r.getID() + r.getItemsRated());
        }
    }

// 2. write a public method named getMovieSize
// returns the number of movies that were read in and stored in the ArrayList of type Movie.
    public int getMovieSize() {return myMovies.size();}


// 3. write a public method named getRaterSize
// returns the number of raters that were read in and stored in the ArrayList of type Rater.
    public int getRaterSize() {return myRaters.size();}

}

```


Create a new class MovieRunnerAverage.


```java

import java.security.Principal;
import java.util.*;
import edu.duke.*;
import org.apache.commons.csv.*;

public class MovieRunnerAverage {


// 1. create a void method named printAverageRatings
// has no parameters.
// Create a SecondRatings object and use the CSV filenames of movie information and ratings information from the first assignment when calling the constructor.
// Print the number of movies and number of raters from the two files by calling the appropriate methods in the SecondRatings class.

    public void printAverageRatings() {
        int minimalRaters = 3;
        // Create a SecondRatings object and use the CSV filenames of movie information and ratings information from the first assignment when calling the constructor.
        SecondRatings sratings = new SecondRatings("ratedmovies_short.csv", "ratings_short.csv");

        // Print the number of movies and number of raters from the two files by calling the appropriate methods in the SecondRatings class.
        System.out.println("Number of total movies: " + sratings.getMovieSize());
        System.out.println("Number of total of Raters: " + sratings.getRaterSize());
    }

}
// Test your program to make sure it is reading in all the data from the two files. (ratings_short.csv and ratedmovies_short.csv, you should see 5 raters and 5 movies.)
```


In the SecondRatings class,

```java

// 1. write a private helper method named getAverageByID
// has two parameters: a String named id for movie ID and an integer named minimalRaters.
// returns a double representing the average movie rating for this ID if there are at least minimalRaters ratings.
// If there are not minimalRaters ratings, then it returns 0.0.

    public double getAverageByID(String id, int minimalRaters) {
        // System.out.println("=======getAverageByID()======");
        double avgRating = 0.0;
        ArrayList<Rater> movieRaterL = new ArrayList<Rater>();
        ArrayList<Double> movieRatingL = new ArrayList<Double>();
        for(Rater i : myRaters){
            if(i.hasRating(id)){
                movieRaterL.add(i);
                movieRatingL.add(i.getRating(id));
            }
        }
        if(movieRaterL.size() > minimalRaters){
            for(Double value : movieRatingL){
                avgRating += value;
            }
            avgRating = avgRating / movieRaterL.size();
        }
        return avgRating;
    }


// 2. write a public method named getAverageRatings
// has one int parameter named minimalRaters.
// find the average rating for every movie that has been rated by at least minimalRaters raters.
// Store each such rating in a Rating object in which the movie ID and the average rating are used in creating the Rating object.
// return an ArrayList of all the Rating objects for movies that have at least the minimal number of raters supplying a rating.
// For example, if minimalRaters has the value 10, then this method returns rating information (the movie ID and its average rating) for each movie that has at least 10 ratings.
// You should consider calling the private getAverageByID method.


    public ArrayList<Rating> getAverageRatings(int minimalRaters) {
        // System.out.println("=======getAverageRatings()======");
        ArrayList<Rating> ratingL = new ArrayList<Rating>();
        for(Movie m : myMovies){
            Double avgRating = getAverageByID(m.getID(), minimalRaters);
            if(avgRating != 0){
                ratingL.add(new Rating(m.getID(), avgRating));
                // System.out.println("movieID: " + m.getID() + " AvgRating: " +avgRating);
            }
        }
        // System.out.println(ratingL.size());
        return ratingL;
    }


// 3.write a method named getTitle
// has one String parameter named id, the ID of a movie.
// returns the title of the movie with that ID.
// If the movie ID does not exist, then this method should return a String indicating the ID was not found.

    public String getTitle(String id) {
        String movieTitle = "";
        for(Movie m : myMovies){
            if(m.getID().equals(id)){
                // System.out.println(movieTitle + m.getTitle());
                return movieTitle + m.getTitle();
            }
        }
        // System.out.println(movieTitle + "not found");
        return movieTitle + "not found";
    }
```



In the MovieRunnerAverage class in the printAverageRatings method

```java

import java.security.Principal;
import java.util.*;
import edu.duke.*;
import org.apache.commons.csv.*;

public class MovieRunnerAverage {

    public void printAverageRatings() {

        int minimalRaters = 3;

        // Create a SecondRatings object and use the CSV filenames of movie information and ratings information from the first assignment when calling the constructor.
        SecondRatings sratings = new SecondRatings("ratedmovies_short.csv", "ratings_short.csv");

        // Print the number of movies and number of raters from the two files by calling the appropriate methods in the SecondRatings class.
        System.out.println("Number of total movies: " + sratings.getMovieSize());
        System.out.println("Number of total of Raters: " + sratings.getRaterSize());


        // add code to print a list of movies and their average ratings, for all those movies that have at least a specified number of ratings, sorted by averages.
        // Specifically, this method should print the list of movies, one movie per line (print its rating first, followed by its title) in sorted order by ratings, lowest rating to highest rating.
        // tset:
        // ratings_short.csv and ratedmovies_short.csv with the argument 3, then the output will display two movies:
        // 8.25 Her
        // 9.0 The Godfather

        ArrayList<Rating> movieAvg = sratings.getAverageRatings(minimalRaters);

        //Method to sort them as requested:
        Collections.sort(movieAvg);
        for(Rating r : movieAvg){
            System.out.println(r);
            String movieID = r.getItem();
            String movieTitle = sratings.getTitle(movieID);
            //We also used a method to round the rating value:
            System.out.println(r.getValue() + " " + movieTitle);
        }
        // System.out.println("Movies with at least " + minimum + " ratings: " + arrayMovies.size());
    }
}
```

---


```java
// In the SecondRatings class,
// write a method getID
// has one String parameter named title, the title of a movie.
// This method returns the movie ID of this movie.
// If the title is not found, return an appropriate message such as “NO SUCH TITLE.”
// Note that the movie title must be spelled exactly as it appears in the movie data files.

    public String getID(String title) {
        for(Movie m : myMovies){
            if(m.getTitle().equals(title)){
                String movieID = m.getID;
                return movieID
            }
        }
        // System.out.println(movieTitle + "not found");
        return "NO SUCH TITLE.";
    }



// In the MovieRunnerAverage class,
// write the void method getAverageRatingOneMovie,
// has no parameters.
// This method should first create a SecondRatings object, reading in data from the movie and ratings data files.
// Then this method should print out the average ratings for a specific movie title, such as the movie “The Godfather”.
// (ratedmovies_short.csv, and ratings_short.csv, then the average for the movie “The Godfather”  would be 9.0.)

// public Rating (String anItem, double aValue)

    public void getAverageRatingOneMovie() {
        // SecondRatings sratings = new SecondRatings("ratedmovies_short.csv", "ratings_short.csv");
        SecondRatings sratings = new SecondRatings(moviefile, ratingfile);

        // String movieTitle = "The Godfather";
        // String movieTitle = "No Country for Old Men";
        // String movieTitle = "Inside Llewyn Davis";
        String movieTitle = "The Maze Runner";
        // String movieTitle = "Vacation";
        String movieID = sratings.getID(movieTitle);

        // Rating (String anItem, double aValue)
        double avgRating = sratings.getAverageByID(movieID, 0);
        System.out.println("the average for the movie:" + movieTitle + " is: " + avgRating);
    }
```



---


.
