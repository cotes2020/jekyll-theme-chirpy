package main

import (
	"errors"
	"fmt"
)

func sayhello(name string, age int) (string, error) {
	result := ""
	if len(name) > 5 {
		return result, errors.New("Your name is too long")
	}
	result = fmt.Sprintf("Hello, %s you are %v years old!\n", name, age)
	return result, nil
}

func main() {
	name := "William"
	age := 21
	message, err := sayhello(name, age)
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf(message)
}
