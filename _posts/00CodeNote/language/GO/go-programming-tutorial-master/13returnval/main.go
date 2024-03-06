package main

import (
	"fmt"
)

func sayhello(name string, age int) string {
	result := fmt.Sprintf("Hello, %s you are %v years old!\n", name, age)
	return result
}

func main() {
	name := "Bill"
	age := 21
	message := sayhello(name, age)
	fmt.Printf(message)
}
