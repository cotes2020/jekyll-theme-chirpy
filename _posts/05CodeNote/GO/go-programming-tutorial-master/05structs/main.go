package main

import (
	"fmt"
)

// Car defines a car
type Car struct {
	wheels int
	color  string
}

func main() {
	var camaro Car
	camaro.wheels = 4
	camaro.color = "Black"
	fmt.Printf("Your Camaro has %v wheels and is %s.", camaro.wheels, camaro.color)
}
