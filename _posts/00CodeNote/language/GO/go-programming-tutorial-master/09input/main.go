package main

import (
	"fmt"
)

func main() {
	var input string
	for {
		fmt.Printf("Enter your name:")
		fmt.Scanln(&input)
		if input == "stop" {
			break
		} else {
			fmt.Println("You Entered:", input)
			fmt.Printf("You Entered: %v", input)
			break
		}
	}
}
