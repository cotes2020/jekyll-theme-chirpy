package randomutil

import (
	uuid "github.com/satori/go.uuid"
	"math/rand"
	"strings"
	"time"
)

// RandomNum 生成 [s,e)区间的一个随机数（注意：不包括e）
func RandomNum(s int64, e int64) int64 {
	//随机数如果 Seed不变 则生成的随机数一直不变
	rand.Seed(time.Now().UnixNano())
	r := rand.Int63n(e - s)
	return s + r
}

// RandomStr 随机生成一个指定长度的uuid
func RandomStr(len int) string {
	nUid := uuid.NewV4().String()
	str := strings.Replace(nUid, "-", "", -1)
	if len < 0 || len >= 32 {
		return str
	}
	return str[:len]
}
