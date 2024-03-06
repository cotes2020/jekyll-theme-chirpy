package main

import (
	"fmt"
)

func main() {
	num := 1
	for num <= 3 {
		fmt.Printf("print line %v\n", num)
		num++
	}
}
