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

# DukeJava 4-3 Step Three Interfaces, Filters, Database - Filtering Recomendations

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 5.Java-Programming-Build-a-Recommendation-System
  - Step One : get the rating, rater, movie from the file
  - Step Two : Simple recommendations
  - Step Three : Interfaces, Filters, Database

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


## Assignment 1: Efficiency

making the program you have already written more efficient.
- start with files from the previous assignment
- make a Rater interface
- and then make a more efficient Rater class.

Specifically for this assignment, you will do the following.



Create a new public interface named `Rater`.

```java
// Add methods to this new interface by copying all the method signatures from the PlainRater class.
// Copy just the methods—do not include the constructors or the private instance variables.

import org.apache.commons.csv.*;
import java.util.*;

public interface Rater {
    public void addRating(String item, double rating);
    public boolean hasRating(String item);
    public String getID();
    public double getRating(String item);
    public int numRatings();
    public ArrayList<String> getItemsRated();
}
```


Change the name of the class `Rater.java` to `PlainRater.java`.

```java
import java.util.*;

public class PlainRater implements Rater{

    private String myID;
    private ArrayList<Rating> myRatings;
    // Rating (String anItem, double aValue)

    public PlainRater(String id) {
        myID = id;
        myRatings = new ArrayList<Rating>();
    }


    public void addRating(String item, double rating) {
        myRatings.add(new Rating(item,rating));
    }

    public boolean hasRating(String item) {
        for(int k=0; k < myRatings.size(); k++){
            if (myRatings.get(k).getItem().equals(item)){
                return true;
            }
        }
        return false;
    }

    public String getID() {
        return myID;
    }

    public double getRating(String item) {
        for(int k=0; k < myRatings.size(); k++){
            if (myRatings.get(k).getItem().equals(item)){
                return myRatings.get(k).getValue();
            }
        }
        return -1;
    }

    public int numRatings() {
        return myRatings.size();
    }

    public ArrayList<String> getItemsRated() {
        ArrayList<String> list = new ArrayList<String>();
        for(int k=0; k < myRatings.size(); k++){
            list.add(myRatings.get(k).getItem());
        }
        return list;
    }

}

```


Create a new class named EfficientRater, and copy the PlainRater class into this class.

```JAVA
// make several changes to this class, including:

import java.util.*;

public class EfficientRater implements Rater{

// Change the ArrayList of type Rating private variable to a HashMap<String,Rating>.
// The key in the HashMap is a movie ID,
// and its value is a rating associated with this movie.
    private String myID;
    private HashMap<String,Rating> myRatings;
    // Rating (String anItem, double aValue)

    public EfficientRater(String id) {
        myID = id;
        myRatings = new HashMap<String,Rating>();
    }

// change addRating to instead add a new Rating to the HashMap with the value associated with the movie ID String item as the key in the HashMap.
    public void addRating(String movieID, double rating) {
        if (!myRatings.containsKey(movieID)) {
            Rating newRating = new Rating(movieID,rating);
            myRatings.put(movieID, newRating);
        }
    }

// The method hasRating should now be much shorter; it no longer needs a loop.
    public boolean hasRating(String movieID) {
        if (myRatings.containsKey(movieID)) {
            return true;
        }
        return false;
    }

    public String getID() {
        return myID;
    }

    public double getRating(String movieID) {
        if (myRatings.containsKey(movieID)) {
            return myRatings.get(movieID).getValue();
        }
        return -1;
    }

    public int numRatings() {
        return myRatings.size();
    }

    public ArrayList<String> getItemsRated() {
        ArrayList<String> list = new ArrayList<String>();
        for(String movieID : myRatings.keySet()){
            list.add(movieID);
        }
        return list;
    }
}
```

Now change FirstRatings to use EfficientRater instead of PlainRater.


```java
import java.security.Principal;
import java.util.*;
import edu.duke.*;
import org.apache.commons.csv.*;

public class FirstRatings{

    private ArrayList<Movie> csvMovieme(CSVParser parser){
        ArrayList<Movie> MovieData = new ArrayList<Movie>();
        try {
            for(CSVRecord movie : parser){
                Movie newMovie = new Movie(movie.get("id"),movie.get("title"),movie.get("year"), movie.get("genre"),movie.get("director"),movie.get("country"),movie.get("poster"),Integer.parseInt(movie.get("minutes").trim()));
                MovieData.add(newMovie);
            }
        } catch(Exception ioe) {
            System.out.println("IOException caught in FirstRatings.csvMovieme");
        }
        return MovieData;
    }

    // loadMovies -> movieData [Movie, Movie, Movie, Movie, ...]
    // id,title,year,country,genre,director,minutes,poster
    // 0006414,"Screen",1916,"USA","Short, Comedy",30,"https://...jpg"
    // return an ArrayList of type Movie with all of the movie data from the file.
    public ArrayList<Movie> loadMovies(String filename){
        FileResource fr = new FileResource("data/" + filename);
        CSVParser parser = fr.getCSVParser();
        ArrayList<Movie> movieData = csvMovieme(parser);
        //Returning an ArrayList of type Movie with data processed from file [filename]
        return movieData;
    }


    // movieData [Movie, Movie, Movie, Movie, ...]
    public void testLoadMovies(){
        // String filename = "ratedmovies_short.csv";
        // String filename = "ratedmoviesfull.csv";
        String filename = "test.movie.csv";
        ArrayList<Movie> movieData = loadMovies(filename);

        System.out.println("the number of movies: " + movieData.size());
        System.out.println("print each movie: ");
        for(Movie i:movieData){
            // System.out.print(i.getTitle() + ", ");
            System.out.println(i.getTitle());
        }
        System.out.println();

        int ccount=0;
        int lcount=0;
        int mNum = 0;
        HashMap<String, Integer> dmap = new HashMap<String, Integer>();
        HashSet<String> directorsWithMaxMovies = new HashSet<String>();
        for(Movie i:movieData){

            // how many movies include the Comedy genre.
            if(i.getGenres().contains("Comedy")){
                ccount+=1;
            }
            // how many movies are greater than 150 minutes in length
            if(i.getMinutes() > 150){
                lcount+=1;
            }

            // determine the maximum number of movies by any director, and who the directors are that directed that many movies.
            String[] directors = i.getDirector().replaceAll(", ", "").split(",");
            for(String dname : directors) {
                if(dmap.containsKey(dname)) {
                    dmap.put(dname, dmap.get(dname) + 1);
                } else {
                    dmap.put(dname, 1);
                }
                mNum = dmap.get(dname) > mNum? dmap.get(dname):mNum;
            }
        }
        for(String dname : dmap.keySet()) {
            if(dmap.get(dname) == mNum) {
                directorsWithMaxMovies.add(dname);
            }
        }
        System.out.println("how many movies include the Comedy genre: " + ccount);
        System.out.println("how many movies are greater than 150 minutes in length: " + lcount);
        System.out.println("the maximum number of movies by any director is " + mNum);

        System.out.println(directorsWithMaxMovies.size() + " directors are directed that many movies are: ");
        for(String dir : directorsWithMaxMovies) {
            // System.out.println(dir + ", ");
            String[] questionL = {"Ridley Scott", "Alfred Hitchcock", "Steven Spielberg", "Woody Allen", "Martin Scorsese"};
            for(String ans : questionL){
                if(ans.equals(dir)){
                    System.out.println(dir);
                    System.out.println(dmap.get(dir));
                }
            }
        }
        System.out.println();
    }


    private ArrayList<Rater> csvRater(CSVParser parser){
        ArrayList<Rater> raterslist = new ArrayList<Rater>();
        for(CSVRecord csvrate : parser){
            Rater newRater = new EfficientRater(csvrate.get("rater_id"));
            newRater.addRating(csvrate.get("movie_id"),Double.parseDouble(csvrate.get("rating")));
            raterslist.add(newRater);
        }
        return raterslist;
    }


    // using EfficientRater:
    // private String myID;
    // private HashMap<String,Rating> myRatings;
    // -> raterslist [Rater, Rater, Rater, ...]
    //                 // rater_id,movie_id,rating,time
    //                 // 1,0068646,10,1381620027
    public ArrayList<Rater> loadRaters(String filename){
        FileResource fr = new FileResource("data/" + filename);
        CSVParser pr = fr.getCSVParser();
        ArrayList<Rater> raterslist = csvRater(pr);
        return raterslist;
    }


    // -> raterData [Rater, Rater, Rater, ...]
    // -> raterList [String: [Rating, Rating, Rating], String: [Rating, Rating, Rating], ...]
                    // rater_id,movie_id,rating,time
                    // 1,0068646,10,1381620027
    public void testLoadRaters(int raterID, String movieID){
        // String filename = "data/ratings_short.csv";
        String filename = "data/ratings.csv";
        ArrayList<Rater> raterData = loadRaters(filename);

        String targetid = Integer.toString(raterID);
        int mNum = 0;

        // Print the total number of raters.
        System.out.println("the total number of raters: " + raterData.size());

        for(Rater i : raterData){

            System.out.println("==============================================");

            String id = i.getID();
            ArrayList<String> itemL = i.getItemsRated();

            // Then for each rater, print the rater’s ID and the number of ratings they did on one line, followed by each rating (both the movie ID and the rating given) on a separate line.
            // System.out.println("the rater’s ID: " + id);
            // System.out.println("    the number of ratings they did: " + itemL.size());
            // for(String item : itemL){
            //     double ratingValue = i.getRating(item);
            //     System.out.print("      movie ID: " + item);
            //     System.out.println(". the rating given: " + ratingValue);
            // }
            // System.out.println("==============================================");


            // Add code to find the number of ratings for a particular rater you specify in your code.
            // For example, if you run this code on the rater whose rater_id is 2 for the file ratings_short.csv, you will see they have three ratings.
            if(id.equals(targetid)){
                System.out.println("the target rater_id: "+ targetid);
                System.out.println("the rater has " + itemL.size() + " ratings: ");
                for(String item : itemL){
                    double ratingValue = i.getRating(item);
                    System.out.print("    movie ID: " + item);
                    System.out.println(". the rating given: " + ratingValue);
                }
            }
            mNum = itemL.size()>mNum? itemL.size():mNum;
        }
        System.out.println("==============================================");


        // Add code to find the maximum number of ratings by any rater. Determine how many raters have this maximum number of ratings and who those raters are.
        // If you run this code on the file ratings_short.csv, you will see rater 2 has three ratings, the maximum number of ratings of all the raters, and that there is only one rater with three ratings.
        System.out.println("the maximum number of ratings by any rateris: "+mNum);
        System.out.print("who those raters are: ");
        for(Rater i : raterData){
            if(i.getItemsRated().size()==mNum){
                System.out.print(i.getID() + ", ");
            }
        }
        System.out.println();
        System.out.println("==============================================");



        // Add code to find the number of ratings a particular movie has.
        // If you run this code on the file ratings_short.csv for the movie “1798709”, you will see it was rated by four raters.
        int ratercount = 0;
        for(Rater i : raterData){
            if(i.hasRating(movieID)){
                ratercount += 1;
            }
        }
        System.out.println("Movie " + movieID + " was rated by " + ratercount + " raters.");
        System.out.println("==============================================");



        // Add code to determine how many different movies have been rated by all these raters.
        // If you run this code on the file ratings_short.csv, you will see there were four movies rated.
        ArrayList<String> movieL = new ArrayList<String>();
        for(Rater i : raterData){
            ArrayList<String> itemL = i.getItemsRated();
            for( String moviename : itemL){
                if(!movieL.contains(moviename)){
                    movieL.add(moviename);
                }
            }
        }
        System.out.println("In " + filename + ": there were " + movieL.size() + " movies rated.");
    }


    public static void main(String[] args) {

        FirstRatings pr = new FirstRatings();

        pr.testLoadMovies();
        // pr.testLoadRaters(193, "1798709");
    }
}
```

---


## Assignment 2


### Additional Starter Files for Assignment 2

The class MovieDatabase

```JAVA
// an efficient way to get information about movies.
// It stores movie information in a HashMap for fast lookup of movie information given a movie ID.
// The class also allows filtering movies based on queries.
// All methods and fields in the class are static.
// This means you'll be able to access methods in MovieDatabase without using new to create objects, but by calling methods like MovieDatabase.getMovie("0120915").

// This class has the following parts:

import java.util.*;
import org.apache.commons.csv.*;
import edu.duke.FileResource;

public class MovieDatabase {

    // - A HashMap named ourMovies
    // maps a movie ID String to a Movie object with all the information about that movie.
    private static HashMap<String, Movie> ourMovies;

    // - A public initialize method
    // with one String parameter named moviefile.
    // call this method with the name of the file used to initialize the movie database.
    public static void initialize(String moviefile) {
        if (ourMovies == null) {
            ourMovies = new HashMap<String,Movie>();
            loadMovies("data/" + moviefile);
        }
    }

    // - A private initialize method
    // with no parameters
    // will load the movie file ratedmoviesfull.csv if no file has been loaded.
    // This method is called as a safety check with any of the other public methods to make sure there is movie data in the database.
    private static void initialize() {
        if (ourMovies == null) {
            ourMovies = new HashMap<String,Movie>();
            loadMovies("data/ratedmoviesfull.csv");
        }
    }

    // - A private loadMovies method
    // build the HashMap.
    private static void loadMovies(String filename) {
        FirstRatings fr = new FirstRatings();
        ArrayList<Movie> list = fr.loadMovies(filename);
        for (Movie m : list) {
            ourMovies.put(m.getID(), m);
        }
    }

    // - A containsID method
    // with one String parameter named id.
    // returns true if the id is a movie in the database, and false otherwise.
    public static boolean containsID(String id) {
        initialize();
        return ourMovies.containsKey(id);
    }

    // - Several getter methods
    // including getYear, getTitle, getMovie, getPoster, getMinutes, getCountry, getGenres, and getDirector.
    // takes a movie ID as a parameter and returns information about that movie.
    public static int getYear(String id) {
        initialize();
        return ourMovies.get(id).getYear();
    }

    public static String getGenres(String id) {
        initialize();
        return ourMovies.get(id).getGenres();
    }

    public static String getTitle(String id) {
        initialize();
        return ourMovies.get(id).getTitle();
    }

    public static Movie getMovie(String id) {
        initialize();
        return ourMovies.get(id);
    }

    public static String getPoster(String id) {
        initialize();
        return ourMovies.get(id).getPoster();
    }

    public static int getMinutes(String id) {
        initialize();
        return ourMovies.get(id).getMinutes();
    }

    public static String getCountry(String id) {
        initialize();
        return ourMovies.get(id).getCountry();
    }

    public static String getDirector(String id) {
        initialize();
        return ourMovies.get(id).getDirector();
    }

    // - A size method
    // returns the number of movies in the database.
    public static int size() {
        return ourMovies.size();
    }


    // - A filterBy method
    // has one Filter parameter named f.
    // returns an ArrayList of type String of movie IDs that match the filtering criteria.
    public static ArrayList<String> filterBy(Filter f) {
        initialize();
        ArrayList<String> list = new ArrayList<String>();
        for(String id : ourMovies.keySet()) {
            if (f.satisfies(id)) {
                list.add(id);
            }
        }
        return list;
    }

}
```

---

### Assignment 2: Filters


### The interface Filter

```java
// has only one signature for the method satisfies. Any filters that implement this interface must also have this method. The method satisfies has one String parameter named id representing a movie ID. This method returns true if the movie satisfies the criteria in the method and returns false otherwise.

public interface Filter {
	public boolean satisfies(String id);
}
```


### The class TrueFilter

```java

// can be used to select every movie from MovieDatabase.
// It’s satisfies method always returns true.
public class TrueFilter implements Filter {
	@Override
	public boolean satisfies(String id) {
		return true;
	}

}
```

### The class AllFilters combines several filters.

```java
import java.util.ArrayList;

public class AllFilters implements Filter {

    // - A private variable named filters that is an ArrayList of type Filter.
    ArrayList<Filter> filters;

    public AllFilters() {
        filters = new ArrayList<Filter>();
    }

    // - An addFilter method
    // has one parameter named f of type Filter.
    // This method allows one to add a Filter to the ArrayList filters.
    public void addFilter(Filter f) {
        filters.add(f);
    }

    // - A satisfies method
    // has one parameter named id, a movie ID.
    // This method returns true if the movie satisfies the criteria of all the filters in the filters ArrayList. Otherwise this method returns false.
    @Override
    public boolean satisfies(String id) {
        for(Filter f : filters) {
            if (! f.satisfies(id)) {
                return false;
            }
        }
        return true;
    }
}
```


#### The class YearsAfterFilter

filter for a specified year;

```java
// it selects only those movies that were created on that year or created later than that year.
// If the year is 2000, then all movies created in the year 2000 and the years after (2001, 2002, 2003, etc) would be selected if used with MovieDatabase.filterBy.

public class YearAfterFilter implements Filter {
	private int myYear;

	public YearAfterFilter(int year) {
		myYear = year;
	}

	@Override
	public boolean satisfies(String id) {
		return MovieDatabase.getYear(id) >= myYear;
	}

}
```


#### GenreFilter

```java
// Create a new class named GenreFilter that implements Filter.
// The constructor should have one parameter named genre representing one genre,
// and the satisfies method should return true if a movie has this genre.
// Note that movies may have several genres.

public class GenreFilter implements Filter {
	private String myGenre;

	public GenreFilter(String genre) {
		myGenre = genre;
		System.out.println("finding Genre: " + myGenre);
	}

	@Override
	public boolean satisfies(String id) {
		// String movieGenre = MovieDatabase.getGenres(id);
		// System.out.println(movieGenre);
		// if(movieGenre.contains(myGenre)){
		// 	System.out.println("with genre");
		// }
		// else {System.out.println("no genre found");}
		// String myStr = "Crime, Drama";
		// System.out.println(myStr.contains("Crime"));

		return MovieDatabase.getGenres(id).contains(myGenre);
	}
}
```


#### MinutesFilter

```java
// Create a new class named MinutesFilter that implements Filter.
// return true if a movie’s running time is at least min minutes and no more than max minutes.

public class MinutesFilter implements Filter {
	private int myMin;
	private int myMax;

	public MinutesFilter(int min, int max) {
		myMin = min;
		myMax = max;
		System.out.println("time period: " + myMin + " - " + myMax);
	}

	@Override
	public boolean satisfies(String id) {
		// String movieGenre = MovieDatabase.getGenres(id);
		// System.out.println(movieGenre);
		// if(movieGenre.contains(myGenre)){
		// 	System.out.println("with genre");
		// }
		// else {System.out.println("no genre found");}
		// String myStr = "Crime, Drama";
		// System.out.println(myStr.contains("Crime"));

		return myMin <= MovieDatabase.getMinutes(id) && MovieDatabase.getMinutes(id) <= myMax;
	}
}
```


#### DirectorsFilter

```java
import java.util.ArrayList;

// Create a new class named DirectorsFilter that implements Filter.
// The constructor should have one parameter named directors representing a list of directors separated by commas (Example: "Charles Chaplin,Michael Mann,Spike Jonze"), and its satisfies method should return true if a movie has at least one of these directors as one of its directors.

public class DirectorsFilter implements Filter {
	private String myDir;

	public DirectorsFilter(String directors) {
		myDir = directors;
	}

	@Override
	public boolean satisfies(String id) {
		String currDir = MovieDatabase.getDirector(id);
		for(String directorName : myDir.split(",")) {
			if(currDir.contains(directorName)) {
				return true;
			}
		}
		return false;
	}
}
```


---

### run file

MovieRunnerWithFilters


```java
// Create a new class named MovieRunnerWithFilters
// use to find the average rating of movies using different filters.
// Copy the printAverageRatings method from the MovieRunnerAverage class into this class.

import java.util.*;

public class MovieRunnerWithFilters {

    String moviefile;
    String ratingfile;
    String datainfoTrueorFalse;
    ThirdRatings tRatings;

    // initialize where to get the movie and rater data,
    // movie.csv + ratings.csv
    public MovieRunnerWithFilters() {
        System.out.println("---------initialize the MovieRunnerWithFilters---------");
        // moviefile = "test.movie.csv";
        // ratingfile = "test.ratings.csv";
        moviefile = "data/ratedmoviesfull.csv";
        ratingfile = "data/ratings.csv";
        // moviefile = "data/ratedmovies_short.csv";
        // ratingfile = "data/ratings_short.csv";
        datainfoTrueorFalse = "True";

        tRatings = new ThirdRatings(ratingfile, datainfoTrueorFalse);
        //   call FirstRatings.loadRaters(ratingsfile) to fill the myRaters ArrayList from ratingfile.
        //        got ArrayList<Rater> myRaters;
        //   if moreinfoTrueorFalse == True
        //        print "Print the Rater detailed info"
        //        print out all the raterID, movieID, and it's Rating under the rater
        System.out.println("Number of total of Raters: " + tRatings.getRaterSize());

        MovieDatabase.initialize(moviefile);
        System.out.println("Number of total of movies: " + MovieDatabase.size());
        //   call FirstRatings.loadMovies(moviefile) to fill the Movies ArrayList from moviefile.
        //        got ArrayList<Movie> list
    }



    public void printAverageRatings(int minRatersNum, String moreinfoTrueorFalse) {
        // setup the minimum Rater Number, and whethere to print out the detailed rater info.
        System.out.println("---------printAverageRatings---------");
        int minimalRaters = minRatersNum;

        // get the average movie rating for this ID if there are at least minimalRaters ratings.
        ArrayList<Rating> avgRatingL = tRatings.getAverageRatings(minimalRaters);
        // Method to sort according the rating
        Collections.sort(avgRatingL);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings");
        System.out.println(" +++++++++ Average Ratings is: " + avgRatingL.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : avgRatingL){
                String movieID = r.getItem();
                String movieTitle = MovieDatabase.getTitle(movieID);
                System.out.println( (double)Math.round(r.getValue()*10000d) / 10000d+ " " + movieTitle);
            }
        }
        //   if moreinfoTrueorFalse == True
        //        print out the movie average rating and movie name
    }


    // create a void method named printAverageRatingsByYear
    // should be similar to printAverageRatings,
    // create a YearAfterFilter and call getAverageRatingsByFilter to get an ArrayList of type Rating of all the movies that have a specified number of minimal ratings and came out in a specified year or later.
    // Print the number of movies found, and for each movie found, print its rating, its year, and its title.
    // For example ratings_short.csv and ratedmovies_short.csv, minimal rater of 1 and the year 2000
    public void printAverageRatingsByYear(int minRatersNum, int yearNum, String moreinfoTrueorFalse) {
        System.out.println("---------printAverageRatingsByYear---------");
        int minimalRaters = minRatersNum;

        YearAfterFilter filterYear = new YearAfterFilter(yearNum);

        ArrayList<Rating> anwList = tRatings.getAverageRatingsByFilter(minimalRaters,filterYear);

        Collections.sort(anwList);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings and By " + yearNum);
        System.out.println(" +++++++++ Movies found: " + anwList.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : anwList){
                String item = r.getItem();
                String movieTitle = MovieDatabase.getTitle(item);
                int movieYear = MovieDatabase.getYear(item);
                System.out.println((double)Math.round(r.getValue() * 10000d) / 10000d + " " + movieYear + " "  + movieTitle);
            }
        }
        //   if moreinfoTrueorFalse == True
        //        print out the movie average rating and movie name
    }


    // create a void method named printAverageRatingsByGenre
    // create a GenreFilter
    // call getAverageRatingsByFilter to get an ArrayList of type Rating of all the movies that have a specified number of minimal ratings and include a specified genre.
    // Print the number of movies found, and for each movie, print its rating and its title on one line, and its genres on the next line.
    // For example, if you run the printAverageRatingsByGenre method on the files ratings_short.csv and ratedmovies_short.csv with a minimal rater of 1 and the genre “Crime”, you should see

    public void printAverageRatingsByGenre(int minRatersNum, String geneWord, String moreinfoTrueorFalse) {
        System.out.println("---------printAverageRatingsByGenre---------");

        int minimalRaters = minRatersNum;

        GenreFilter filterGenre = new GenreFilter(geneWord);

        ArrayList<Rating> anwList = tRatings.getAverageRatingsByFilter(minimalRaters, filterGenre);
        Collections.sort(anwList);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings and Genre as " + geneWord);
        System.out.println(" +++++++++ Movies found: " + anwList.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : anwList){
                String movieID = r.getItem();
                String movieTitle = MovieDatabase.getTitle(movieID);
                String movieGenre = MovieDatabase.getGenres(movieID);
                System.out.println((double)Math.round(r.getValue() * 10000d) / 10000d + " " + movieTitle);
                System.out.println("     " + movieGenre);
            }
        }
    }


    // create a void method named printAverageRatingsByMinutes
    // create a MinutesFilter and call getAverageRatingsByFilter to get an ArrayList of type Rating of all the movies that have a specified number of minimal ratings and their running time is at least a minimum number of minutes and no more than a maximum number of minutes.
    // Print the number of movies found, and for each movie print its rating, its running time, and its title on one line.
    // For example
    // ratings_short.csv and ratedmovies_short.csv, minimal rater of 1, minimum minutes of 110, and maximum minutes of 170
    public void printAverageRatingsByMinutes(int minRatersNum, int minMin, int maxMin, String moreinfoTrueorFalse) {
        System.out.println("---------printAverageRatingsByMinutes---------");
        int minimalRaters = minRatersNum;
        MinutesFilter filterMin = new MinutesFilter(minMin, maxMin);

        ArrayList<Rating> anwList = tRatings.getAverageRatingsByFilter(minimalRaters, filterMin);
        Collections.sort(anwList);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings and duration between " + minMin + " and " + maxMin);
        System.out.println(" +++++++++ Movies found: " + anwList.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : anwList){
                String movieID = r.getItem();
                String movieTitle = MovieDatabase.getTitle(movieID);
                int movieMin = MovieDatabase.getMinutes(movieID);
                System.out.println((double)Math.round(r.getValue() * 10000d) / 10000d + " Time:" + movieMin + " " + movieTitle);
            }
        }
    }


    // create a void method named printAverageRatingsByDirectors
    // create a DirectorsFilter
    // call getAverageRatingsByFilter to get an ArrayList of type Rating of all the movies that have a specified number of minimal ratings and include at least one of the directors specified.
    // Print the number of movies found, and for each movie print its rating and its title on one line, and all its directors on the next line.
    // For example, ratings_short.csv and ratedmovies_short.csv with a minimal rater of 1 and the directors set to "Charles Chaplin,Michael Mann,Spike Jonze"
    public void printAverageRatingsByDirectors(int minRatersNum, String dirName, String moreinfoTrueorFalse) {
        System.out.println("---------printAverageRatingsByDirectors---------");
        int minimalRaters = minRatersNum;
        String targetDir = dirName;

        DirectorsFilter filterDir = new DirectorsFilter(targetDir);

        ArrayList<Rating> anwList = tRatings.getAverageRatingsByFilter(minimalRaters, filterDir);
        Collections.sort(anwList);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings and by directory " + dirName);
        System.out.println(" +++++++++ Movies found: " + anwList.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : anwList){
                String movieID = r.getItem();
                String movieTitle = MovieDatabase.getTitle(movieID);
                String movieDirs = MovieDatabase.getDirector(movieID);
                System.out.println((double)Math.round(r.getValue() * 10000d) / 10000d + " " + movieTitle);

                for(String directorName : targetDir.split(",")) {
                    if(movieDirs.contains(directorName)) {
                        System.out.println("    " + directorName);
                        break;
                    }
                }
            }
        }
    }




    // create a void method named printAverageRatingsByYearAfterAndGenre
    // create an AllFilters object that includes criteria based on movies that came out in a specified year or later and have a specified genre as one of its genres.
    // call getAverageRatingsByFilter to get an ArrayList of type Rating of all the movies that have a specified number of minimal ratings and the two criteria based on year and genre.
    // Print the number of movies found, and for each movie, print its rating, its year, and its title on one line, and all its genres on the next line.
    // For example ratings_short.csv and ratedmovies_short.csv
    // minimal rater of 1, the year set to 1980, and the genre set to “Romance”
    // minimal rater of 1, minimum minutes set to 30, maximum minutes set to 170, and the directors set to "Spike Jonze,Michael Mann,Charles Chaplin,Francis Ford Coppola"
    public void printAverageRatingsByYearAfterAndGenre(int minRatersNum, String moreinfoTrueorFalse) {
        System.out.println("---------printAverageRatingsByYearAfterAndGenre---------");
        int minimalRaters = minRatersNum;
        int targetYear = 0;
        String targetGen = "null";
        String targetDir = "null";
        int minMin = 0;
        int maxMin = 0;

        AllFilters allFilters = new AllFilters();

        // targetYear = 1990;
        // YearAfterFilter filterYear = new YearAfterFilter(targetYear);

        // targetGen = "Drama";
        // GenreFilter filterGen = new GenreFilter(targetGen);

        targetDir = "Clint Eastwood,Joel Coen,Tim Burton,Ron Howard,Nora Ephron,Sydney Pollack";
        DirectorsFilter filterDir = new DirectorsFilter(targetDir);

        minMin = 90;
        maxMin = 180;
        MinutesFilter filterMin = new MinutesFilter(minMin, maxMin);

        // allFilters.addFilter(filterGen);
        // allFilters.addFilter(filterYear);
        allFilters.addFilter(filterDir);
        allFilters.addFilter(filterMin);

        ArrayList<Rating> anwList = tRatings.getAverageRatingsByFilter(minimalRaters, allFilters);
        Collections.sort(anwList);

        System.out.println(" +++++++++ Movies with at least " + minimalRaters + " ratings and by filter:");
        System.out.println(" +++++++++ Year: " + targetYear);
        System.out.println(" +++++++++ Gene: " + targetGen);
        System.out.println(" +++++++++ Director: " + targetDir);
        System.out.println(" +++++++++ Duration: " + minMin + ", " + maxMin);
        System.out.println(" +++++++++ Movies found: " + anwList.size());

        if(moreinfoTrueorFalse.equals("True")){
            for(Rating r : anwList){
                String movieID = r.getItem();
                String movieTitle = MovieDatabase.getTitle(movieID);
                String movieGenre = MovieDatabase.getGenres(movieID);
                String movieDirs = MovieDatabase.getDirector(movieID);
                int movieYear = MovieDatabase.getYear(movieID);
                int movieMin = MovieDatabase.getMinutes(movieID);

                System.out.println((double)Math.round(r.getValue() * 10000d) / 10000d + " Time:" + movieMin + " Year: " + movieYear + " "  + " Title: " + movieTitle);
                System.out.println("     Genere: " + movieGenre);

                for(String directorName : targetDir.split(",")) {
                    if(movieDirs.contains(directorName)) {
                        System.out.println("    director: " + directorName);
                        break;
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        MovieRunnerWithFilters pr = new MovieRunnerWithFilters();

        // test result:
        // pr.printAverageRatings(1, "1True");
        // pr.printAverageRatingsByYear(1, 2000, "1True");
        // pr.printAverageRatingsByGenre(1, "Crime", "1True");
        // pr.printAverageRatingsByMinutes(1, 110, 170, "1True");
        // pr.printAverageRatingsByDirectors(1, "Charles Chaplin,Michael Mann,Spike Jonze", "1True");
        // pr.printAverageRatingsByYearAfterAndGenre(1, "1True");

        // Quiz:
        pr.printAverageRatings(35, "1True");
        pr.printAverageRatingsByYear(20, 2000, "1True");
        pr.printAverageRatingsByGenre(20, "Comedy", "1True");
        pr.printAverageRatingsByMinutes(5, 105, 135, "1True");
        pr.printAverageRatingsByDirectors(4, "Clint Eastwood,Joel Coen,Martin Scorsese,Roman Polanski,Nora Ephron,Ridley Scott,Sydney Pollack", "1True");
        pr.printAverageRatingsByYearAfterAndGenre(3, "1True");
    }
}

// Number of total of Raters: 1048
// Number of total of movies: 3143
// ---------printAverageRatings---------
//  +++++++++ Movies with at least 35 ratings
//  +++++++++ Average Ratings is: 29
// ---------printAverageRatingsByYear---------
//  +++++++++ Movies with at least 20 ratings and By 2000
//  +++++++++ Movies found: 88
// ---------printAverageRatingsByGenre---------
// finding Genre: Comedy
//  +++++++++ Movies with at least 20 ratings and Genre as Comedy
//  +++++++++ Movies found: 19
// ---------printAverageRatingsByMinutes---------
// time period: 105 - 135
//  +++++++++ Movies with at least 5 ratings and duration between 105 and 135
//  +++++++++ Movies found: 231
// ---------printAverageRatingsByDirectors---------
//  +++++++++ Movies with at least 4 ratings and by directory Clint Eastwood,Joel Coen,Martin Scorsese,Roman Polanski,Nora Ephron,Ridley Scott,Sydney Pollack
//  +++++++++ Movies found: 22
// ---------printAverageRatingsByYearAfterAndGenre---------
// finding Genre: Drama
// time period: 30 - 170
//  +++++++++ Movies with at least 8 ratings and by filter:
//  +++++++++ Year: 1990
//  +++++++++ Gene: Drama
//  +++++++++ Director: Spike Jonze,Michael Mann,Charles Chaplin,Francis Ford Coppola
//  +++++++++ Duration: 30, 170
//  +++++++++ Movies found: 132
// ---------printAverageRatingsByYearAfterAndGenre---------
// time period: 90 - 180
//  +++++++++ Movies with at least 3 ratings and by filter:
//  +++++++++ Year: 0
//  +++++++++ Gene: null
//  +++++++++ Director: Clint Eastwood,Joel Coen,Tim Burton,Ron Howard,Nora Ephron,Sydney Pollack
//  +++++++++ Duration: 90, 180
//  +++++++++ Movies found: 15
```



.
