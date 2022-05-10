package main

import (
	"fmt"
)

func sayhello(name string, age int) {
	fmt.Printf("Hello, %s you are %v years old!\n", name, age)
}

func main() {
	name := "Bill"
	age := 21
	sayhello(name, age)
}
