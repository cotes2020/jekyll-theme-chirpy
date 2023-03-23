package main

import (
	"fmt"
)

var (
	name = []string{"Bill", "Ted", "Frank"}
)

const (
	pi = 3.14
)

func main() {
	fmt.Printf("Hello, %s %v !\n", name[1:2], pi)
	fmt.Printf("Hello, %s %v !\n", name[:2], pi)
	fmt.Printf("Hello, %s %v !\n", name, pi)

	name = append(name, "Jenny")
	fmt.Printf("Hello, %s %v !\n", name, pi)
}
