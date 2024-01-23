package main

import (
	"fmt"
)

func main() {

	for x := 1; x < 4; x++ {
		fmt.Printf("print a line %v\n", x)
	}

	nums := []int{1, 2, 3}

	for x := range nums {
		fmt.Printf("print line %v\n", x)
	}

	//  for index, value := range nums {
	for x, y := range nums {
		fmt.Printf("print index %v, num %v\n", x, y)
	}

	for _, y := range nums {
		fmt.Printf("print value %v\n", y)
	}
}
