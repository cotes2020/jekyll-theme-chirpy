---
title: Java - DukeJava - 4-1-1 - Programming Exercise - Searching Earthquake Data
date: 2020-09-10 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---


# DukeJava - 4-1-1 - Programming Exercise - Searching Earthquake Data

[toc]

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 4.Java-Programming-Principles-of-Software-Design
  - 4-1-1 Programming Exercise 1 - Searching Earthquake Data

[Resource Link](https://www.dukelearntoprogram.com/course4/index.php)

[ProjectCode](https://github.com/ocholuo/language/tree/master/0.project/javademo)

---


The classes provided are:

- The class Location, from the Android platform and revised for this course, a data class representing a geographic location. One of the constructors has parameters latitude and longitude, and one of the public methods is distanceTo.


- The class QuakeEntry
  - has a `constructor` that requires `latitude, longitude, magnitude, title, and depth`.
  - It has several `get` methods and a `toString` method.

```java
public class QuakeEntry implements Comparable<QuakeEntry> {

    private Location myLocation;
    private String title;
    private double depth;
    private double magnitude;

    public QuakeEntry(double lat, double lon, double mag, String t, double d) {
        myLocation = new Location(lat,lon);
        magnitude = mag;
        title = t;
        depth = d;
    }

    public Location getLocation(){return myLocation;}

    public double getMagnitude(){return magnitude;}

    public String getInfo(){return title;}

    public double getDepth(){return depth;}

    @Override
    public int compareTo(QuakeEntry loc) {
        double difflat = myLocation.getLatitude() - loc.myLocation.getLatitude();
        if (Math.abs(difflat) < 0.001) {
            double diff = myLocation.getLongitude() - loc.myLocation.getLongitude();
            if (diff < 0) return -1;
            if (diff > 0) return 1;
            return 0;
        }
        if (difflat < 0) return -1;
        if (difflat > 0) return 1;

        // never reached
        return 0;
    }

    public String toString(){
        return String.format("(%3.2f, %3.2f), mag = %3.2f, depth = %3.2f, title = %s", myLocation.getLatitude(),myLocation.getLongitude(),magnitude,depth,title);
    }

}
```

- The class `EarthQuakeParser`
  - has a `read` method with one String parameter, represents an XML earthquake data file and returns an `ArrayList of QuakeEntry objects`.

```java
import java.io.File;
import java.io.IOException;
import java.util.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.*;
import org.xml.sax.SAXException;


public class EarthQuakeParser {

    public EarthQuakeParser() {
        // TODO Auto-generated constructor stub
    }

    public ArrayList<QuakeEntry> read(String source) {

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();

        try {
            DocumentBuilder builder = factory.newDocumentBuilder();

            //Document document = builder.parse(new File(source));
            //Document document = builder.parse("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.atom");
            Document document = null;

            if (source.startsWith("http")){document = builder.parse(source);}
            else {document = builder.parse(new File(source));}
            //Document document = builder.parse("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom");

            NodeList nodeList = document.getDocumentElement().getChildNodes();

            ArrayList<QuakeEntry> list = new ArrayList<QuakeEntry>();

            for(int k=0; k < nodeList.getLength(); k++){
                Node node = nodeList.item(k);

                if (node.getNodeName().equals("entry")) {
                    Element elem = (Element) node;
                    NodeList t1 = elem.getElementsByTagName("georss:point");
                    NodeList t2 = elem.getElementsByTagName("title");
                    NodeList t3 = elem.getElementsByTagName("georss:elev");
                    double lat = 0.0, lon = 0.0, depth = 0.0;
                    String title = "NO INFORMATION";
                    double mag = 0.0;

                    if (t1 != null) {
                        String s2 = t1.item(0).getChildNodes().item(0).getNodeValue();
                        //System.out.print("point2: "+s2);
                        String[] args = s2.split(" ");
                        lat = Double.parseDouble(args[0]);
                        lon = Double.parseDouble(args[1]);
                    }
                    if (t2 != null){
                        String s2 = t2.item(0).getChildNodes().item(0).getNodeValue();

                        String mags = s2.substring(2,s2.indexOf(" ",2));
                        if (mags.contains("?")) {
                            mag = 0.0;
                            System.err.println("unknown magnitude in data");
                        }
                        else {
                            mag = Double.parseDouble(mags);
                            //System.out.println("mag= "+mag);
                        }
                        int sp = s2.indexOf(" ",5);
                        title = s2.substring(sp+1);
                        if (title.startsWith("-")){
                            int pos = title.indexOf(" ");
                            title = title.substring(pos+1);
                        }
                    }
                    if (t3 != null){
                        String s2 = t3.item(0).getChildNodes().item(0).getNodeValue();
                        depth = Double.parseDouble(s2);
                    }
                    QuakeEntry loc = new QuakeEntry(lat,lon,mag,title,depth);
                    list.add(loc);
                }

            }
            return list;
        }
        catch (ParserConfigurationException pce){System.err.println("parser configuration exception");}
        catch (SAXException se){System.err.println("sax exception");}
        catch (IOException ioe){System.err.println("ioexception");}
        return null;
    }

    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException{
        EarthQuakeParser xp = new EarthQuakeParser();
        //String source = "data/2.5_week.atom";
        //String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        String source = "data/nov20quakedata.atom";
        ArrayList<QuakeEntry> list  = xp.read(source);
        Collections.sort(list);
        for(QuakeEntry loc : list){
            System.out.println(loc);
        }
        System.out.println("# quakes = "+list.size());

    }

}
```


- The class EarthQuakeClient
  - has been started for you and creates an EarthQuakeParser to read in an earthquake data file, creating an ArrayList of QuakeEntrys.
  - You can test the program with the method `createCSV` to store an ArrayList of the earthquake data and print a CSV file.
  - You will complete the methods that filter magnitude and distance in this class and add additional methods to it.

```java
import java.util.*;
import edu.duke.*;

public class EarthQuakeClient {

    private EarthQuakeParser parser;
    private String source;
    private ArrayList<QuakeEntry> quakeData;


    public EarthQuakeClient() {
        parser = new EarthQuakeParser();
        source = "data/nov20quakedatasmallsmall.atom";
        // source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        quakeData  = parser.read(source);
    }


    public void dumpCSV(ArrayList<QuakeEntry> list){
        System.out.println("Latitude,Longitude,Magnitude,Info");
        for(QuakeEntry qe : list){
            System.out.printf("%4.2f,%4.2f,%4.2f,%s\n",
                qe.getLocation().getLatitude(),
                qe.getLocation().getLongitude(),
                qe.getMagnitude(),
                qe.getInfo());
        }

    }
```


- The class ClosestQuakes
  - to find the ten closest quakes to a particular location.
  - You will complete this method.

```java

```



---

## Assignment 1: Filtering by Magnitude and Distance

Modify the `EarthQuakeClient` class:


```java
// 1. Write the method `filterByMagnitude`:
// two parameters, an ArrayList of type QuakeEntry named quakeData, and a double named magMin.
// return an ArrayList of type QuakeEntry of all the earthquakes from quakeData that have a magnitude larger than magMin.
// Notice that we have already created an ArrayList named answer for you to store those earthquakes that satisfy this requirement.

    // ------------------------------------------------------------
    // ArrayList<QuakeEntry> list {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // for each QuakeEntry if QuakeEntry.getMagnitude() > magMin
    // return ArrayList<QuakeEntry>
    // ------------------------------------------------------------
    // This method should return an ArrayList of type QuakeEntry of all the earthquakes from quakeData that have a magnitude larger than magMin.
    public ArrayList<QuakeEntry> filterByMagnitude(ArrayList<QuakeEntry> quakeData, double magMin) {
        ArrayList<QuakeEntry> answer = new ArrayList<QuakeEntry>();
        for(QuakeEntry qe : quakeData){
            if(qe.getMagnitude() > magMin){
                answer.add(qe);
            }
        }
        return answer;
    }



// 2. Modify the method `bigQuakes`
// no parameters
// use filterByMagnitude and print earthquakes above a certain magnitude, and the number of such earthquakes.
// this method reads data on earthquakes from a file, stores a QuakeEntry for each earthquake read in the ArrayList named list, and prints out the number of earthquakes read in.

    // ------------------------------------------------------------
    // EarthQuakeParser.read(source) = ArrayList<QuakeEntry> list {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // for each QuakeEntry return QuakeEntry.getMagnitude() > 5.0
    // ------------------------------------------------------------
    // this method reads data on earthquakes from a file, stores a QuakeEntry for each earthquake read in the ArrayList named list, and prints out the number of earthquakes read in.
    public void bigQuakes() {
        EarthQuakeParser parser = new EarthQuakeParser();
        //String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        String source = "data/nov20quakedatasmall.atom";
        ArrayList<QuakeEntry> list  = parser.read(source);
        System.out.println("read data for "+list.size()+" quakes");

        ArrayList<QuakeEntry> listBig = filterByMagnitude(list, 5.0);
        for(QuakeEntry qe : listBig){
            System.out.println(qe);
        }
    }



// 3. Write the method `filterByDistanceFrom`
// has three parameters, an ArrayList of type QuakeEntry named quakeData, a double named distMax, and a Location named from.
// return an ArrayList of type QuakeEntry of all the earthquakes from quakeData that are less than distMax from the location from.

    // ------------------------------------------------------------
    // ArrayList<QuakeEntry> quakeData {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // call filterByDistance to calculate each QuakeEntry distance,
    // print out the earthquakes within 1000 Kilometers to a specified city
    // ------------------------------------------------------------
    public ArrayList<QuakeEntry> filterByDistanceFrom(ArrayList<QuakeEntry> quakeData, double distMax, Location from) {
        ArrayList<QuakeEntry> answer = new ArrayList<QuakeEntry>();
        for(QuakeEntry qe : quakeData){
            Location loc = qe.getLocation();
            double dis = loc.distanceTo(from);
            if(dis < distMax){
                answer.add(qe);
            }
        }
        // System.out.println(answer.size() + " QuakeEntry has been founded.");
        return answer;
    }




// 4. Modify the method `closeToMe`
// has no parameters
// call `filterByDistance` to print out the earthquakes within 1000 Kilometers to a specified city (such as Durham, NC).
// For each earthquake found, print the distance from the earthquake to the specified city, followed by the information about the city (use getInfo()).
// Currently this method reads data on earthquakes from a URL, stores a QuakeEntry for each earthquake read in the ArrayList named list, and prints out the number of earthquakes read in. It also gives the location for two cities, Durham, NC (35.988, -78.907) and Bridgeport, CA (38.17, -118.82).
// After making modifications, when you run your program on the file `nov20quakedatasmall.atom` for the city location `Durham, NC`, no earthquakes are found. But if you then run the program for the city location `Bridgeport, CA`, seven earthquakes are found.

    // ------------------------------------------------------------
    // EarthQuakeParser.read(source) = ArrayList<QuakeEntry> list {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // call filterByDistance to calculate each QuakeEntry distance,
    // print out the earthquakes within 1000 Kilometers to a specified city
    // ------------------------------------------------------------
    // For each earthquake found, print the distance from the earthquake to the specified city, followed by the information about the city (use getInfo()
    public void closeToMe(){
        EarthQuakeParser parser = new EarthQuakeParser();
        // String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        String source = "data/nov20quakedatasmall.atom";
        ArrayList<QuakeEntry> list  = parser.read(source);
        System.out.println("read data for "+list.size()+" quakes");

        // System.out.println("This location is Durham, NC");
        // Location city = new Location(35.988, -78.907);
        System.out.println("This location is Bridgeport, CA");
        Location city =  new Location(38.17, -118.82);

        ArrayList<QuakeEntry> qeCloseToCity = filterByDistanceFrom(list, 1000000, city);
        for(QuakeEntry qe : qeCloseToCity){
            System.out.println(qe);
            System.out.println((qe.getLocation().distanceTo(city) / 1000) + " Kilometers, " + qe.getInfo());
        }
        System.out.println("Found " + qeCloseToCity.size() + " that match that criteria");
    }

```



---

## Assignment 2: Filtering by Depth



```java
// 1. Write the method `filterByDepth`
// has three parameters, an ArrayList of type QuakeEntry named quakeData, a double named minDepth, a double named maxDepth.
// return an ArrayList of type QuakeEntry of all the earthquakes from quakeData whose depth is between minDepth and maxDepth, exclusive.
// (Do not include quakes with depth exactly minDepth or maxDepth.)

    // ------------------------------------------------------------
    // EarthQuakeParser.read(source) = ArrayList<QuakeEntry> list {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // call filterByDepth to calculate each QuakeEntry Depth
    // ------------------------------------------------------------
    public ArrayList<QuakeEntry> filterByDepth(ArrayList<QuakeEntry> quakeData, double minDepth, double maxDepth) {
        ArrayList<QuakeEntry> qeByDepth = new ArrayList<QuakeEntry>();
        for(QuakeEntry qe : quakeData){
            double dep = qe.getDepth();
            if(minDepth < dep && dep < maxDepth){
                qeByDepth.add(qe);
                System.out.println(qe);
            }
        }
        return qeByDepth;
    }



// 2. Write the void method `quakesOfDepth`
// has no parameters
// use `filterByDepth` to print all the earthquakes from a data source whose depth is between a given minimum and maximum value, and the number of earthquakes found.
// After writing this method, run program on the file `nov20quakedatasmall.atom` for quakes with depth between `-10000.0` and `-5000.0`

    // ------------------------------------------------------------
    // EarthQuakeParser.read(source) = ArrayList<QuakeEntry> list {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // call filterByDepth to calculate each QuakeEntry Depth,
    // print out the earthquakes within Depth range
    // ------------------------------------------------------------
    public void quakesOfDepth() {
        EarthQuakeParser parser = new EarthQuakeParser();
        // String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        String source = "data/nov20quakedata.atom";
        ArrayList<QuakeEntry> quakeData  = parser.read(source);
        System.out.println("read data for "+ quakeData.size()+" quakes");

        ArrayList<QuakeEntry> qeByDepth = filterByDepth(quakeData, -12000.0, -10000.0);
        for(QuakeEntry qe : qeByDepth){
            System.out.println(qe);
        }
        System.out.println("the number of earthquakes found: " + qeByDepth.size());
    }

```
---

## Assignment 3: Filtering by Phrase in Title

add new methods to one class, the EarthQuakeClient class:

```java

// 1. Write the method `filterByPhrase`
// has three parameters:
//     an ArrayList of type QuakeEntry named quakeData,
//     a String named where, that indicates where to search in the title and has one of three values: (“start”, ”end”, or “any”),
//     a String named phrase, indicating the phrase to search for in the title of the earthquake.
// The title of the earthquake can be obtained through the `getInfo()` method.
// return an ArrayList of type QuakeEntry of all the earthquakes from quakeData whose titles have the given phrase found at location where (“start” means the phrase must start the title, “end” means the phrase must end the title and “any” means the phrase is a substring anywhere in the title.)

    public ArrayList<QuakeEntry> filterByPhrase(ArrayList<QuakeEntry> quakeData, String where, String phrase) {
        ArrayList<QuakeEntry> qeByPhrase = new ArrayList<QuakeEntry>();
        for (QuakeEntry qe : quakeData) {
            if (where.equals("start")) {
                if (qe.getInfo().startsWith(phrase)) {
                    qeByPhrase.add(qe);
                }
            } else if (where.equals("end")) {
                if (qe.getInfo().endsWith(phrase)) {
                    qeByPhrase.add(qe);
                }
            } else if (where.equals("any")) {
                if (qe.getInfo().contains(phrase)) {
                    qeByPhrase.add(qe);
                }
            }
        }
        return qeByPhrase;
    }



// 2. Write the void method `quakesByPhrase`
// use `filterByPhrase` to print all the earthquakes from a data source that have phrase in their title in a given position in the title, and the number of earthquakes found.
// run on the file `nov20quakedatasmall`.atom for quakes
// with phrase `“California”` and where set to `“end”`
// with phrase “Can” and where set to “any”
// with phrase “Explosion” and where set to “start”

    public void quakesByPhrase() {
        EarthQuakeParser parser = new EarthQuakeParser();
        // String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        String source = "data/nov20quakedata.atom";
        ArrayList<QuakeEntry> quakeData  = parser.read(source);
        System.out.println("read data for "+ quakeData.size()+" quakes");

        // String where = "end";
        // String phrase = "California";
        // ArrayList<QuakeEntry> qeByPhrase = filterByPhrase(quakeData, where, phrase);
        // for(QuakeEntry qe : qeByPhrase){
        //     System.out.println(qe);
        // }
        // System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);


        // String where = "any";
        // String phrase = "Creek";
        // ArrayList<QuakeEntry> qeByPhrase = filterByPhrase(quakeData, where, phrase);
        // for(QuakeEntry qe : qeByPhrase){
        //     System.out.println(qe);
        // }
        // System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);


        // String where = "start";
        // String phrase = "Explosion";
        // ArrayList<QuakeEntry> qeByPhrase = filterByPhrase(quakeData, where, phrase);
        // for(QuakeEntry qe : qeByPhrase){
        //     System.out.println(qe);
        // }
        // System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);

        String where = "start";
        String phrase = "Quarry Blast";
        ArrayList<QuakeEntry> qeByPhrase = filterByPhrase(quakeData, where, phrase);
        for(QuakeEntry qe : qeByPhrase){
            System.out.println(qe);
        }
        System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);

        where = "end";
        phrase = "Alaska";
        qeByPhrase = filterByPhrase(quakeData, where, phrase);
        for(QuakeEntry qe : qeByPhrase){
            System.out.println(qe);
        }
        System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);


        where = "any";
        phrase = "Can";
        qeByPhrase = filterByPhrase(quakeData, where, phrase);
        for(QuakeEntry qe : qeByPhrase){
            System.out.println(qe);
        }
        System.out.println("Found " + qeByPhrase.size() + " quakes that match " + phrase + " at " + where);
    }

```


---

## Assignment 4: Finding the Closest Earthquakes to a Location


Modify the `ClosestQuakes` class:


```java

// 1. The method `findClosestQuakes`
// reads in data on earthquakes storing them in the ArrayList list, prints how many quakes there are.
// It sets a location variable named jakarta to the location of the city Jakarta.
// It then calls the method getClosest to determine the ten closest earthquakes in list and prints information about those quakes and how close they are to Jakarta.

    // ------------------------------------------------------------
    // get ArrayList<QuakeEntry> quakeData {QuakeEntry1, QuakeEntry2, QuakeEntry3, ...}
    // return howMany quake = find closest to the location, remove it, find next, loop 10 times
    // ------------------------------------------------------------
    public ArrayList<QuakeEntry> getClosest(ArrayList<QuakeEntry> quakeData, Location current, int howMany) {
        ArrayList<QuakeEntry> copy = new ArrayList<QuakeEntry>(quakeData);
        ArrayList<QuakeEntry> closestL = new ArrayList<QuakeEntry>();
        for(int i = 0; i < howMany; i++){
            int minIndex = 0;
            double minDistance = Integer.MAX_VALUE;

            for(int k = 0; k< copy.size(); k++){
                // QuakeEntry qe = copy.get(k);
                // Location loc = qe.getLocation();
                double currDistance = copy.get(k).getLocation().distanceTo(current);
                // Location minloc = copy.get(minIndex).getLocation();
                if(currDistance < minDistance){
                    minIndex = k;
                }
            }
            closestL.add(copy.get(minIndex));
            copy.remove(minIndex);
        }
        return closestL;
    }



// 2. Complete the method `getClosest`.
// has three parameters,
//     an ArrayList of type QuakeEntry named quakeData,
//     a Location named current,
//     and an int named howMany.
// find the closest number of howMany earthquakes to the current Location, return them in an ArrayList of type QuakeEntry.
// The earthquakes should be in the ArrayList in order with the closest earthquake in index position 0.
// If there are fewer then howMany earthquakes in quakeData, then the ArrayList returned would be the same size as quakeData.

    public void findClosestQuakes() {
        EarthQuakeParser parser = new EarthQuakeParser();
        String source = "data/nov20quakedatasmall.atom";
        // String source = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.atom";
        ArrayList<QuakeEntry> list  = parser.read(source);
        System.out.println("read data for "+list.size());

        Location jakarta  = new Location(-6.211,106.845);

        ArrayList<QuakeEntry> close = getClosest(list,jakarta,3);
        for(int k=0; k < close.size(); k++){
            QuakeEntry entry = close.get(k);
            double distanceInMeters = jakarta.distanceTo(entry.getLocation());
            System.out.printf("%4.2f\t %s\n", distanceInMeters/1000,entry);
        }
        System.out.println("number found: "+close.size());
    }


// Now run the method findClosestQuakes by calling getClosest with the location current set to Jakarta (-6.211,106.845) and howMany set to 3. When you run your program on the file nov20quakedatasmall.atom
// read data for 25
// 14534.43	 (34.05, -117.36), mag = 1.20, depth = 1040.00, title = Quarry Blast - 4km WNW of Grand Terrace, California
// 8439.34	 (-24.67, -175.93), mag = 5.10, depth = -10000.00, title = South of Tonga
// 6153.14	 (38.27, 142.53), mag = 4.60, depth = -30500.00, title = 109km E of Ishinomaki, Japan
// number found: 3
```

---

## Assignment 5: Finding the Largest Magnitude Earthquakes


Write a new class named `LargestQuakes`.
- to determine the N biggest earthquakes, those with largest magnitude.

```java
import java.util.*;


public class LargestQuakes {

// 1. Write a void method named `findLargestQuakes`
// reads in earthquake data from a source and storing them into an ArrayList of type QuakeEntry.
// it prints all the earthquakes and how many earthquakes that were from the source.
// read in earthquakes from the small file nov20quakedatasmall.atom, print all the earthquakes and also print how many there are.
// comment out the printing of all the earthquakes, but continue to print out the total number of earthquakes read in.

    public ArrayList<QuakeEntry> findLargestQuakes(int howMany) {
        EarthQuakeParser parser = new EarthQuakeParser();
        String source = "data/nov20quakedata.atom";
        ArrayList<QuakeEntry> quakeData  = parser.read(source);

        ArrayList<QuakeEntry> answer = getLargest(quakeData, howMany);
        int ind = 0;
        for(QuakeEntry qe : answer){
            ind++;
            System.out.println(ind + ".  " + qe);
        }
        System.out.println("total number of earthquakes read in: " + answer.size());
        return answer;
    }


// 2. Write a method named `indexOfLargest`
// has one parameter, an ArrayList of type QuakeEntry named data.
// returns an integer representing the index location in data of the earthquake with the largest magnitude.
// test out this method by adding code to the method `findLargestQuakes` to print the index location of the largest magnitude earthquake in the file nov20quakedatasmall.atom and the earthquake at that location.
// will see that the largest such earthquake is at location 3 and has magnitude 5.50.

    public int indexOfLargest(ArrayList<QuakeEntry> quakeData) {
        int largestIndex = 0;
        double largestMagnitude = 0.0;
        for(int i = 0; i < quakeData.size(); i++){
            QuakeEntry currQe = quakeData.get(i);
            double currMagnitude = currQe.getMagnitude();
            if(largestMagnitude < currMagnitude){
                largestMagnitude = currMagnitude;
                largestIndex = i;
            }
        }
        System.out.println("the largest such earthquake is at location " + largestIndex + ", magnitude: " + largestMagnitude);
        return largestIndex;
    }


// 3. Write a method named `getLargest`
// has two parameters, an ArrayList of type QuakeEntry named quakeData, an integer named howMany.
// returns an ArrayList of type QuakeEntry of the top howMany largest magnitude earthquakes from quakeData.
// The quakes returned should be in the ArrayList in order by their magnitude, with the largest magnitude earthquake in index position 0.
// If quakeData has fewer than howMany earthquakes, then the number of earthquakes returned in the ArrayList is equal to the number of earthquakes in quakeData.
// This method should call the method `indexOfLargest`.

    public ArrayList<QuakeEntry> getLargest(ArrayList<QuakeEntry> quakeData, int howMany) {
        ArrayList<QuakeEntry> copy = new ArrayList<QuakeEntry>(quakeData);
        ArrayList<QuakeEntry> answer = new ArrayList<QuakeEntry>();
        for(int i = 0; i < howMany; i++){
            int largestIndex = indexOfLargest(copy);
            answer.add(copy.get(largestIndex));
            copy.remove(largestIndex);
        }
        return answer;
    }

//  Modify the method findLargestQuakes to call getLargest to print the 5 earthquakes of largest magnitude from the file nov20quakedatasmall.atom. Those five earthquakes are:
// the largest such earthquake is at location 3 and has magnitude 5.5
// the largest such earthquake is at location 11 and has magnitude 5.1
// the largest such earthquake is at location 21 and has magnitude 5.1
// the largest such earthquake is at location 15 and has magnitude 5.0
// the largest such earthquake is at location 4 and has magnitude 4.9
// (26.38, 142.71), mag = 5.50, depth = -12890.00, title = 91km SSE of Chichi-shima, Japan
// (-11.63, 165.52), mag = 5.10, depth = -20700.00, title = 106km SSW of Lata, Solomon Islands
// (-24.67, -175.93), mag = 5.10, depth = -10000.00, title = South of Tonga
// (8.53, -71.34), mag = 5.00, depth = -25160.00, title = 5km ENE of Lagunillas, Venezuela
// (40.37, 73.20), mag = 4.90, depth = -40790.00, title = 21km WNW of Gulcha, Kyrgyzstan
// total number of earthquakes read in: 5
```









.
