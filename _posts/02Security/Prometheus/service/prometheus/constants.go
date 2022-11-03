package prometheus

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/ocholuo/ocholuo.github.io/_posts/02Security/Prometheus/utils/randomutil"
)

const (
	namePrefix = "the_number_of_student"
	subSys     = "client_golang"
	nameSpace  = "prometheus_demo"
)

var names = []string{"小明", "小红", "小花"}
var ages = []int64{20, 21, 22, 23, 24, 25}
var subjects = []string{"语文", "数学", "体育"}

// get num int from http.Request
func GetParamNum(req *http.Request) int64 {
	err := req.ParseForm()
	if err != nil {
		fmt.Println("parse form err")
		return 0
	}
	numStr := req.Form.Get("num")
	fmt.Printf("numStr:[%v]\n", numStr)
	num, err := strconv.ParseInt(numStr, 10, 64)
	if err != nil {
		fmt.Printf("parse int err :%v\n", err)
		return 0
	}
	return num
}

// randData随机获取一个元素，并且将本次请求的随机元素统计到countStrMap
func getCurRandomStrMap(countStrMap map[string]int64, randData []string) string {
	index := randomutil.RandomNum(0, int64(len(randData)))
	randVal := randData[index]
	countStrMap[randVal] = countStrMap[randVal] + 1
	return randVal
}

// randData随机获取一个元素，并且将本次请求的随机元素统计到countIntMap
func getCurRandomIntMap(countIntMap map[int64]int64, randData []int64) string {
	index := randomutil.RandomNum(0, int64(len(randData)))
	randVal := randData[index]
	countIntMap[randVal] = countIntMap[randVal] + 1
	return fmt.Sprintf("%d", randVal)
}
