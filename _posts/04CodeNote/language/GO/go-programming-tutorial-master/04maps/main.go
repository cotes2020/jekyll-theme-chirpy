package main

import (
	"fmt"
)

var (
	level = map[string]int{
		"Bill":  25,
		"Ted":   1,
		"Frank": 8,
	}
)

func main() {
	fmt.Printf("Hello, Bill you are level %v !\n", level["Bill"])
}
