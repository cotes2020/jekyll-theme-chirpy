---
title: Java - DukeJava 4-3 Step Three Interfaces, Filters, Database
date: 2020-09-14 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

[toc]

---

# DukeJava 4-4 Step Four Calculating Weighted Averages

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 5.Java-Programming-Build-a-Recommendation-System
  - Step One : get the rating, rater, movie from the file
  - Step Two : Simple recommendations
  - Step Three : Interfaces, Filters, Database
  - Step Four : Calculating Weighted Averages

Resource Link: https://www.coursera.org/learn/java-programming-recommender/supplement/KTrOQ/programming-exercise-step-two

starter files: https://www.dukelearntoprogram.com//course5/files.php

---


```java
Movie (String anID, String aTitle, String aYear, String theGenres, String aDirector, String aCountry, String aPoster, int theMinutes)

Rater(String id, ArrayList<Rating>())

EfficientRater (String id, HashMap<String,Rating>)

Rating (String anItem, double aValue)

MovieDatabase ( HashMap<String movieID, Movie m> ourMovies)

// ---------------------------------

public interface Rater {
    public void addRating(String item, double rating);
    public boolean hasRating(String item);
    public String getID();
    public double getRating(String item);
    public int numRatings();
    public ArrayList<String> getItemsRated();
    public void printRatingHash();
}

public class PlainRater implements Rater{}

public class EfficientRater implements Rater{}

// ---------------------------------

public class MovieDatabase {
    private static HashMap<String, Movie> ourMovies;
    private static void initialize() {
        if (ourMovies == null) {
            ourMovies = new HashMap<String,Movie>();
            loadMovies("data/ratedmoviesfull.csv");
        }
    }
    public static HashMap<String, Movie> initialize(String moviefile){}
    private static void loadMovies(String filename) {
        FirstRatings fr = new FirstRatings();
        ArrayList<Movie> list = fr.loadMovies(filename);
        for (Movie m : list) {ourMovies.put(m.getID(), m);}
    }
    public static boolean containsID(String id){}
    public static int getYear(String id){}
    public static String getGenres(String id){}
    public static String getTitle(String id){}
    public static Movie getMovie(String id){}
    public static String getPoster(String id){}
    public static int getMinutes(String id){}
    public static String getCountry(String id){}
    public static String getDirector(String id){}
    public static int size(){}
    public static ArrayList<String> filterBy(Filter f){}
}

// ---------------------------------

public interface Filter {
	public boolean satisfies(String id);
}

public class AllFilters implements Filter {
    ArrayList<Filter> filters;
    public AllFilters() {filters = new ArrayList<Filter>();}
    public void addFilter(Filter f) {filters.add(f);}
    @Override
    public boolean satisfies(String id){}
}

public class TrueFilter implements Filter {
	@Override
	public boolean satisfies(String id) {return true;}
}

public class YearAfterFilter implements Filter {
	private int myYear;
	public YearAfterFilter(int year){}
	@Override
	public boolean satisfies(String id){}
}

// ---------------------------------

public class FirstRatings{
    private ArrayList<Movie> csvMovieme(CSVParser parser){}
    public ArrayList<Movie> loadMovies(String filename){}
    public void testLoadMovies(){}
    private ArrayList<Rater> csvRater(CSVParser parser){}
    public ArrayList<Rater> loadRaters(String filename){}
    public void testLoadRaters(int raterID, String movieID){}
}

public class SecondRatings {
    private ArrayList<Movie> myMovies;
    private ArrayList<Rater> myRaters;
    public SecondRatings(){}
    public SecondRatings(String moviefile, String ratingsfile){}
    public int getMovieSize() {return myMovies.size();}
    public int getRaterSize() {return myRaters.size();}
    public double getAverageByID(String id, int minimalRaters){}
    public ArrayList<Rating> getAverageRatings(int minimalRaters){}
    public String getTitle(String id){}
    public String getID(String title){}
}

public class ThirdRatings {
    private ArrayList<Rater> myRaters;
    public ThirdRatings(){}
    public ThirdRatings(String ratingsfile){}
    public int getRaterSize() {return myRaters.size();}
    public double getAverageByID(String id, int minimalRaters){}
    public ArrayList<Rating> getAverageRatings(int minimalRaters){}
    public ArrayList<Rating> getAverageRatingsByFilter(int minimalRaters, Filter filterCriteria){}
}

// ---------------------------------

public class MovieRunnerAverage {
    public void printAverageRatings(){}
    public void getAverageRatingOneMovie(){}
}

public class MovieRunnerWithFilters {
    private int helperMoviesAndRatings() {}
    public void printAverageRatings(){}
    public void printAverageRatingsByYear() {}
    public void printAverageRatingsByGenre() {}
    public void printAverageRatingsByMinutes() {}
}

```

---

Calculating Weighted Averages:

![Averages](https://i.imgur.com/VmEDxUo.png)


Calculating closeness:

![closeness](https://i.imgur.com/H2LLwaj.png)



Calculating close:

![close](https://i.imgur.com/tEYvEQO.png)


---


```

13

81

The method getSimilarRatings can be written with just one line of code that returns the result  of a call to getSimilarRatingsByFilter.


---------initialize the MovieRunnerSimilarRatings---------
Number of total of Raters: 1048
Number of total of movies: 3143

 ------------------ getSimilarRatings() ------------------
The movie returned with the top rated average is: Frozen, weighted is: 212.75

 ---------------- printSimilarRatingsByGenre ---------------- xxx
finding Genre: Mystery
The movie returned with the top rated average is: Divergent, weighted is: 467.0


 ------------------ getSimilarRatings() ------------------ xxx
The movie returned with the top rated average is: Boyhood, weighted is: 1565.375

 ---------------- printSimilarRatingsByDirector ----------------
The movie returned with the top rated average is: Star Trek, weighted is: 646.5

 ---------------- printSimilarRatingsByGenreAndMinutes ----------------
finding Genre: Drama
time period: 80 - 160
The movie returned with the top rated average is: The Imitation Game, weighted is: 1204.25

 ---------------- printSimilarRatingsByYearAfterAndMinutes ----------------
time period: 70 - 200
The movie returned with the top rated average is: Nightcrawler, weighted is: 828.125

```





.
