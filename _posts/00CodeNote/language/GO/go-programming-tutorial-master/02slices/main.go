package main

import (
	"fmt"
)

// var name = [1]string

var (
	names = []string{"Bill", "Ted", "Frank"}
)

const (
	pi = 3.14
)

func main() {
	fmt.Printf("Hello, %s %v !\n", names[1:2], pi)
	names[0] = "Bob"
	fmt.Printf("Hello, %s %v !\n", names[0], pi)
}
