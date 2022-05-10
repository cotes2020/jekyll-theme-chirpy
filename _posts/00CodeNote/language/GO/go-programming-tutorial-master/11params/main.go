package main

import (
	"fmt"
)

func sayhello(name string) {
	fmt.Printf("Hello, %s!\n", name)
}

func main() {
	name := "Bill"
	sayhello(name)
}
