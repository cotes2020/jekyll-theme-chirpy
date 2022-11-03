package prometheus

import (
	"fmt"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/ocholuo/ocholuo.github.io/_posts/02Security/Prometheus/utils/randomutil"
)

const counterNamePrefix = namePrefix + "_counter"

// CommonCounter 普通的计数器
var commonCounterTotalCount int64

var CommonCounter = prometheus.NewCounter(
	prometheus.CounterOpts{
		Subsystem: subSys,
		Namespace: nameSpace,
		Help:      "desc the metric",
		Name:      fmt.Sprintf("%s:%s", counterNamePrefix, "common"),
		//ConstLabels: map[string]string{"name":"555"}, // 每个打点必定会带这个label
	},
)

// 有自己的计算函数的计数器，但是要保证递增，只有在客户端来拉去这个指标的时候才会触发这个函数计算获取最新的指标值
var funcCounterTotalCount int64

var FuncCounter = prometheus.NewCounterFunc(
	prometheus.CounterOpts{
		Subsystem: subSys,
		Namespace: nameSpace,
		Name:      fmt.Sprintf("%s:%s", counterNamePrefix, "func"),
	}, func() float64 {
		delta := randomutil.RandomNum(0, 3) // 模拟随机增长步长 0|1|2
		newNum := atomic.AddInt64(&funcCounterTotalCount, delta)
		return float64(newNum)
	})

// VecCounter 带有 "name", "age" 标签的计数器
var vecCounterTotalCount int64

var VecCounter = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: subSys,
		Namespace: nameSpace,
		Name:      fmt.Sprintf("%s:%s", counterNamePrefix, "vec"),
	}, []string{"name", "age"})

func DealCommCounter(w http.ResponseWriter, req *http.Request) {
	dealCount := GetParamNum(req)
	var curDealCount int64

	go func() {
		// 每次打点处理都是定时器3秒一次
		ticker := time.NewTicker(3 * time.Second)
		for {
			<-ticker.C
			// Inc increments the counter by 1.
			// Use Add to increment it by arbitrary non-negative values.
			CommonCounter.Inc()
			curDealCount++
			atomic.AddInt64(&commonCounterTotalCount, 1)

			fmt.Printf(
				"commonCounterTotalCount:%v,curDealCount:%v\n",
				commonCounterTotalCount, curDealCount)

			if curDealCount == dealCount {
				fmt.Println("DealCounter结束")
				return
			}
		}
	}()

	fmt.Fprintf(w, "DealCommCounter done !!!")
}

func DealVecCounter(w http.ResponseWriter, req *http.Request) {
	dealCount := GetParamNum(req)
	var curDealCount int64

	go func() {
		// 每次打点处理都是定时器3秒一次
		ticker := time.NewTicker(3 * time.Second)
		thisNameMap := make(map[string]int64)
		thisAgeMap := make(map[int64]int64)

		for {
			<-ticker.C
			nameStr := getCurRandomStrMap(thisNameMap, names)
			ageStr := getCurRandomIntMap(thisAgeMap, ages)

			VecCounter.With(prometheus.Labels{"name": nameStr, "age": ageStr}).Inc()
			curDealCount++
			atomic.AddInt64(&vecCounterTotalCount, 1)

			fmt.Printf(
				"vecCounterTotalCount:%v,curDealCount:%v, nameMap:%v, ageMap:%v\n",
				vecCounterTotalCount, curDealCount, thisNameMap, thisAgeMap)

			if curDealCount == dealCount {
				fmt.Println("DealVecCounter结束")
				return
			}
		}
	}()

	fmt.Fprintf(w, "DealVecCounter done !!!")
}
