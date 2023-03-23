package main

import (
	"fmt"
)

func main() {
	level := 25
	var response string
	if level > 20 && level < 30 {
		response = "High level"
	} else if level <= 20 {
		response = "Low level"
	} else {
		response = "Super High level"
	}

	fmt.Printf("Your level is %s.", response)
}
