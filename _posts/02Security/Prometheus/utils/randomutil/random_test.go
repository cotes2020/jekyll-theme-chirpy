package randomutil

import (
	"log"
	"testing"
)

func TestRandomNum(t *testing.T) {
	num := RandomNum(29, 30)
	log.Printf("随机数 => [%v]\n", num)
}

func TestRandomStr(t *testing.T) {
	str := RandomStr(31)
	log.Printf("随机字符串 => [%v] len:%v\n", str, len(str))
}
