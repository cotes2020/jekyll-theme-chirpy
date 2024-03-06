package JAVA.javaCrush;

// implements : bring new interface

public class Student implements Comparable<Student>{

    private String name;

    public Student(String name) {
        this.name = name;
    }


    public boolean equals(Student other) {
        if (this.name == other.name) {
            return true;
        }
        else {
            return false;
        }
    }


    public int compareTo(Student other) {
        return this.name.compareTo(other.name);
        // compare how far 2 are away
    }


    public String toString() {
        return this.name;
        // return "Student (" + this.name + ")";
    }


}
