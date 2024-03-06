package main

import (
	"fmt"
)

var (
	name string
)

func main() {
	const pi = 3.14
	name = "Bill"
	fmt.Printf("Hello, %s %v !\n", name, pi)
}
