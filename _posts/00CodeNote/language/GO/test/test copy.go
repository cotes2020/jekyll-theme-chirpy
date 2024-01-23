package main

import (
	"encoding/json"
	"fmt"
)

type Employee struct {
	FirstName string `json:"firstname"`
	LastName  string `json:"lastname"`
	City      string `json:"city"`
}

func main() {
	json_string := `
    {
        "firstname": "Rocky",
        "lastname": "String",
        "city": "London"
    }`

	emp1 := new(Employee)
	json.Unmarshal([]byte(json_string), emp1)
	fmt.Println(emp1)
	// &{Rocky String London}

	emp2 := new(Employee)
	emp2.FirstName = "Ramesh"
	emp2.LastName = "Soni"
	emp2.City = "Mumbai"

	jsonStr, _ := json.Marshal(emp2)
	fmt.Printf("%s\n", jsonStr)
	// {"firstname":"Ramesh","lastname":"Soni","city":"Mumbai"}
}
