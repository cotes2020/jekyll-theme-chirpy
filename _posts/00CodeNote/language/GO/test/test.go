package main

import (
	"flag"
	"fmt"
	"os"
)

var name = flag.String("name", "Tom", "Input your name")
var age = flag.Int("age", 18, "Input your age")
var f = flag.Bool("isVIP", false, "Is VIP")
var postCode int

func init() {
	flag.IntVar(&postCode, "postcode", 1234, "Input your post code")
}

func main() {
	//接受命令行参数
	flag.Parse()

	fmt.Println("name:", *name)
	fmt.Println("age:", *age)
	fmt.Println("VIP:", *f)
	fmt.Println("postCode:", postCode)

	//返回没有被解析的命令行参数
	fmt.Println("reduntant tail:", flag.Args())

	//返回没有被解析的命令行参数的个数
	fmt.Println(flag.NArg())

	//命令行设置的参数个数
	fmt.Println(flag.NFlag())

	//return input 参数
	args := os.Args
	fmt.Println("Args:", args)

	paramCnt := flag.NArg()
	for cnt := 0; cnt < paramCnt; cnt++ {
		fmt.Println(flag.Arg(cnt))
	}
}
