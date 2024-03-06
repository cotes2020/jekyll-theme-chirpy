---
title: Java - DukeJava 4-1 StepOne
date: 2020-09-14 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---

[toc]

---

# DukeJava 4-1 StepOne

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 5.Java-Programming-Build-a-Recommendation-System
  - Step One : get the rating, rater, movie from the file

Resource Link: https://www.coursera.org/learn/java-programming-recommender/supplement/KTrOQ/programming-exercise-step-two

---

The class <kbd>Movie</kbd>
- a Plain Old Java Object (POJO) class for storing the data about one movie.
- It includes the following items:
  - Eight `private variables` to represent information about a movie
  - A `constructor` with eight parameters to initialize the private variables
  - Eight `getter` methods to return the private information
    - such as the method `getGenres`: returns a String of all the genres for this movie.
  - A `toString` method: represent movie information as a String for easy print.

```java
// id - a String variable representing the IMDB ID of the movie
// title - a String variable for the movie’s title
// year - an integer representing the year
// genres - one String of one or more genres separated by commas
// director - one String of one or more directors of the movie separated by commas
// country - one String of one or more countries the film was made in, separated by commas
// minutes - an integer for the length of the movie
// poster - a String that is a link to an image of the movie poster if one exists, or “N/A” if no poster exists

public class Movie {

    private String id;
    private String title;
    private int year;
    private String genres;
    private String director;
    private String country;
    private String poster;
    private int minutes;

    public Movie (String anID, String aTitle, String aYear, String theGenres) {
        // just in case data file contains extra whitespace
        id = anID.trim();
        title = aTitle.trim();
        year = Integer.parseInt(aYear.trim());
        genres = theGenres;
    }

    public Movie (String anID, String aTitle, String aYear, String theGenres, String aDirector,
    String aCountry, String aPoster, int theMinutes) {
        // just in case data file contains extra whitespace
        id = anID.trim();
        title = aTitle.trim();
        year = Integer.parseInt(aYear.trim());
        genres = theGenres;
        director = aDirector;
        country = aCountry;
        poster = aPoster;
        minutes = theMinutes;
    }

    // Returns ID associated with this item
    public String getID () {return id;}
    // Returns title of this item
    public String getTitle () {return title;}
    // Returns year in which this item was published
    public int getYear () {return year;}
    // Returns genres associated with this item
    public String getGenres () {return genres;}
    public String getCountry(){return country;}
    public String getDirector(){return director;}
    public String getPoster(){return poster;}
    public int getMinutes(){return minutes;}

    // Returns a string of the item's information
    public String toString () {
        String result = "Movie [id=" + id + ", title=" + title + ", year=" + year;
        result += ", genres= " + genres + "]";
        return result;
    }
}
```

The class <kbd>Rating</kbd>
- POJO class for storing the data about one rating of an item.
- It includes
  - Two `private variables` to represent information about a rating:
  - A `constructor` with two parameters to initialize the private variables.
  - Two `getter` methods
    - `getItem`
    - `getValue`
  - A `toString` method: represent rating information as a String.
  - A `compareTo` method: compare this rating with another rating.


```java
// item - a String description of the item being rated (for this assignment you should use the IMDB ID of the movie being rated)
// value - a double of the actual rating

public class Rating implements Comparable<Rating> {
    private String item;
    private double value;

    public Rating (String anItem, double aValue) {
        item = anItem;
        value = aValue;
    }

    // Returns item being rated
    public String getItem () {return item;}
    // Returns the value of this rating (as a number so it can be used in calculations)
    public double getValue () {return value;}

    // Returns a string of all the rating information
    public String toString () {return "[" + getItem() + ", " + getValue() + "]";}

    public int compareTo(Rating other) {
        if (value < other.value) return -1;
        if (value > other.value) return 1;
        return 0;
    }
}
```


The class <kbd>Rater</kbd>
- keeps track of one rater and all their ratings.
- This class includes:
  - Two `private variables`:
  - A `constructor` with one parameter of the ID for the rater.
  - A method `addRating`
    - two parameters, a String named item, a double named rating.
    - A new Rating is created and added to myRatings.
  - A method `getID`
    - no parameters
    - get the ID of the rater.
  - A method `getRating`
    - one parameter item.
    - returns the double rating of this item if it is in myRatings.
    - Otherwise this method returns -1.
  - A method `numRatings`
    - returns the number of ratings this rater has.
  - A method `getItemsRated`
    - no parameters.
    - returns an ArrayList of Strings representing a list of all the items that have been rated.


```java
// myID - a unique String ID for this rater
// myRatings - an ArrayList of Ratings

import java.util.*;

public class Rater {

    private String myID;
    private ArrayList<Rating> myRatings;
    // Rating (String anItem, double aValue)
    // {Rating1, Rating2, Rating3}

    public Rater(String id) {
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

    public String getID() {return myID;}

    public double getRating(String item) {
        for(int k=0; k < myRatings.size(); k++){
            if (myRatings.get(k).getItem().equals(item)){
                return myRatings.get(k).getValue();
            }
        }
        return -1;
    }

    public int numRatings() {return myRatings.size();}

    public ArrayList<String> getItemsRated() {
        ArrayList<String> list = new ArrayList<String>();
        for(int k=0; k < myRatings.size(); k++){
            list.add(myRatings.get(k).getItem());
        }
        return list;
    }

}
```

---

ratedmovies_short.csv

```c
id,title,year,country,genre,director,minutes,poster
0006414,"Behind the Screen",1916,"USA","Short, Comedy, Romance","Charles Chaplin",30,"https://ia.media-imdb.com/images/M/MV5BMTkyNDYyNTczNF5BMl5BanBnXkFtZTgwMDU2MzAwMzE@._V1_SX300.jpg"
0068646,"The Godfather",1972,"USA","Crime, Drama","Francis Ford Coppola",175,"https://ia.media-imdb.com/images/M/MV5BMjEyMjcyNDI4MF5BMl5BanBnXkFtZTcwMDA5Mzg3OA@@._V1_SX300.jpg"
0113277,"Heat",1995,"USA","Action, Crime, Drama","Michael Mann",170,"https://ia.media-imdb.com/images/M/MV5BMTM1NDc4ODkxNV5BMl5BanBnXkFtZTcwNTI4ODE3MQ@@._V1_SX300.jpg"
1798709,"Her",2013,"USA","Drama, Romance, Sci-Fi","Spike Jonze",126,"https://ia.media-imdb.com/images/M/MV5BMjA1Nzk0OTM2OF5BMl5BanBnXkFtZTgwNjU2NjEwMDE@._V1_SX300.jpg"
0790636,"Dallas Buyers Club",2013,"USA","Biography, Drama","Jean-Marc VallÃ©e",117,"N/A"
```

ratings_short.csv:

```c
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
```


---

## Assignment

create a new class <kbd>FirstRatings</kbd>
- to process the movie and ratings data and to answer questions about them.
- use CSVParser and CSVRecord.



```java
import java.security.Principal;
import java.util.*;
import edu.duke.*;
import org.apache.commons.csv.*;

public class FirstRatings{


// 1. Write a method named loadMovies
// has one parameter, a String named filename.
// process every record from the CSV file whose name is filename, a file of movie information, and return an ArrayList of type Movie with all of the movie data from the file.

    // loadMovies -> movieData [Movie, Movie, Movie, Movie, ...]
    // id,title,year,country,genre,director,minutes,poster
    // 0006414,"Screen",1916,"USA","Short, Comedy",30,"https://...jpg"
    // return an ArrayList of type Movie with all of the movie data from the file.

    public ArrayList<Movie> loadMovies(String filename){
        FileResource fr = new FileResource("data/" + filename);
        CSVParser parser = fr.getCSVParser();
        ArrayList<Movie> movieData = new ArrayList<Movie>();

        try {
            for(CSVRecord csvr : parser.getRecords()) {
                //Making a temp movie variable to make a Movie Object with data in csvr, this will be added to movieData
                Movie temp = new Movie(csvr.get("id"), csvr.get("title"), csvr.get("year"), csvr.get("genre"),
                    csvr.get("director"), csvr.get("country"), csvr.get("poster"), Integer.parseInt(csvr.get("minutes").trim()));
                //Adding temp to movieData
                movieData.add(temp);
                // System.out.println(temp);
            }
        }
        catch(Exception ioe) {
            System.out.println("IOException caught");
        }
        //Returning an ArrayList of type Movie with data processed from file [filename]return movieDa;
    }


// 2. Write a void method named testLoadMovies
// Call the method loadMovies on the file and store the result in an ArrayList local variable. Print the number of movies, and print each movie.
// 5 movies in ratedmovies_short.csv .
// 3143 movies in ratedmoviesfull.csv.
// Add code to determine how many movies include the Comedy genre. (ratedmovies_short.csv, there is only one.)
// Add code to determine how many movies are greater than 150 minutes in length. (ratedmovies_short.csv, there are two.)
// Add code to determine the maximum number of movies by any director, and who the directors are that directed that many movies. some movies may have more than one director. (ratedmovies_short.csv the maximum number of movies by any director is one, and there are five directors that directed one such movie.)

    // movieData [Movie, Movie, Movie, Movie, ...]
    public void testLoadMovies(){
        // String filename = "ratedmovies_short.csv";
        String filename = "ratedmoviesfull.csv";
        ArrayList<Movie> movieData = loadMovies(filename);
        System.out.println("the number of movies: " + movieData.size());
        System.out.println("print each movie: ");
        for(Movie i:movieData){
            System.out.print(i.getTitle() + ", ");
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


// 3. write a method named loadRaters
// has one parameter named filename.
// process every record from the CSV file whose name is filename, a file of raters and their ratings
// return an ArrayList of type Rater with all the rater data from the file.

    // -> raterData [Rater, Rater, Rater, ...]
    // -> raterList [String: [Rating, Rating, Rating], String: [Rating, Rating, Rating], ...]
                    // rater_id,movie_id,rating,time
                    // 1,0068646,10,1381620027

    public ArrayList<Rater> loadRaters(String filename){
        FileResource fr = new FileResource(filename);
        CSVParser pr = fr.getCSVParser();

        ArrayList<Rater> raterData = new ArrayList<Rater>();
        HashMap<String, ArrayList<Rating>> raterList = new HashMap<String, ArrayList<Rating>>();

        try {
            for(CSVRecord csvr : pr.getRecords()){
                String ratingItem = csvr.get("movie_id");
                double ratingValue = Double.parseDouble(csvr.get("rating"));
                String raterID = csvr.get("rater_id");
                Rating tempRating = new Rating(ratingItem, ratingValue);

                // System.out.println(raterID);
                if(raterList.containsKey(raterID)){
                    // System.out.println("has");
                    raterList.get(raterID).add(tempRating);
                }
                else{
                    // System.out.println("dont have");
                    raterList.put(raterID, new ArrayList<Rating>());
                    raterList.get(raterID).add(tempRating);
                }
            }

            for(String rname : raterList.keySet()){
                Rater r = new Rater(rname);
                for(Rating values : raterList.get(rname)){
                    r.addRating(values.getItem(), values.getValue());
                }
                raterData.add(r);
            }
        }
        catch(Exception ioe) {
            System.out.println("IOException caught");
        }return raterData;}




// Write a void method named testLoadRaters
// Call the method loadRaters on the file ratings_short.csv and store the result in a local ArrayList variable. Print the total number of raters. Then for each rater, print the rater’s ID and the number of ratings they did on one line, followed by each rating (both the movie ID and the rating given) on a separate line.
// (ratings_short.csv, five raters; ratings.csv, 1048 raters)
// Add code to find the number of ratings for a particular rater you specify in your code. For example, if you run this code on the rater whose rater_id is 2 for the file ratings_short.csv, you will see they have three ratings.

// Add code to find the maximum number of ratings by any rater. Determine how many raters have this maximum number of ratings and who those raters are. If you run this code on the file ratings_short.csv, you will see rater 2 has three ratings, the maximum number of ratings of all the raters, and that there is only one rater with three ratings.

// Add code to find the number of ratings a particular movie has. If you run this code on the file ratings_short.csv for the movie “1798709”, you will see it was rated by four raters.

// Add code to determine how many different movies have been rated by all these raters. If you run this code on the file ratings_short.csv, you will see there were four movies rated.

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
        pr.testLoadRaters(193, "1798709");
    }

}
```







.
