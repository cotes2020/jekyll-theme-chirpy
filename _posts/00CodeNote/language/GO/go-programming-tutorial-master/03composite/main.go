package main

import (
	"fmt"
)

var (
	name [1]string
)

const (
	pi = 3.14
)

func main() {
	name[0] = "Bill"
	fmt.Printf("Hello, %s %v !\n", name[0], pi)
}
