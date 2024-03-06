package main

import (
	"log"
	"math/rand"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// 采集器Collector接口的实现
// type Collector interface {
// 	// 用于传递所有可能的指标的定义描述符
// 	// 可以在程序运行期间添加新的描述，收集新的指标信息
// 	// 重复的描述符将被忽略。两个不同的Collector不要设置相同的描述符
// 	Describe(chan<- *Desc)

// 	// Prometheus的注册器调用Collect执行实际的抓取参数的工作，
// 	// 并将收集的数据传递到Channel中返回
// 	// 收集的指标信息来自于Describe中传递，可以并发的执行抓取工作，但是必须要保证线程的安全。
// 	Collect(chan<- Metric)
// }

// 了解了接口的实现后，就可以写自己的实现了
// 先定义 结构体
// 这是一个集群的指标采集器，每个集群都有自己的Zone, 代表集群的名称。
// 另外两个是保存的采集的指标。
type ClusterManager struct {
	Zone         string
	OOMCountDesc *prometheus.Desc
	RAMUsageDesc *prometheus.Desc
}

func NewClusterManager(zone string) *ClusterManager {
	return &ClusterManager{
		Zone: zone,
		OOMCountDesc: prometheus.NewDesc(
			"clustermanager_oom_crashes_total",
			"Number of OOM crashes.",
			[]string{"host"},
			prometheus.Labels{"zone": zone},
		),
		RAMUsageDesc: prometheus.NewDesc(
			"clustermanager_ram_usage_bytes",
			"RAM usage as reported to the cluster manager.",
			[]string{"host"},
			prometheus.Labels{"zone": zone},
		),
	}
}

func (c *ClusterManager) ReallyExpensiveAssessmentOfTheSystemState() (
	oomCountByHost map[string]int, ramUsageByHost map[string]float64,
) {
	oomCountByHost = map[string]int{
		"foo.example.org": int(rand.Int31n(1000)),
		"bar.example.org": int(rand.Int31n(1000)),
	}
	ramUsageByHost = map[string]float64{
		"foo.example.org": rand.Float64() * 100,
		"bar.example.org": rand.Float64() * 100,
	}
	return
}

func (c *ClusterManager) Collect(ch chan<- prometheus.Metric) {
	oomCountByHost, ramUsageByHost := c.ReallyExpensiveAssessmentOfTheSystemState()
	for host, oomCount := range oomCountByHost {
		ch <- prometheus.MustNewConstMetric(
			c.OOMCountDesc,          // 指标描述符
			prometheus.CounterValue, // 指标类型
			float64(oomCount), host,
		)
	}
	for host, ramUsage := range ramUsageByHost {
		ch <- prometheus.MustNewConstMetric(
			c.RAMUsageDesc,
			prometheus.GaugeValue,
			ramUsage, host,
		)
	}
}

func (c *ClusterManager) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.OOMCountDesc
	ch <- c.RAMUsageDesc
}

func main() {

	workerDB := NewClusterManager("db")
	workerCA := NewClusterManager("ca")

	reg := prometheus.NewPedanticRegistry()
	reg.MustRegister(workerDB)
	reg.MustRegister(workerCA)

	gatherers := prometheus.Gatherers{
		// prometheus.DefaultGatherer,
		reg,
	}

	h := promhttp.HandlerFor(
		gatherers,
		promhttp.HandlerOpts{
			ErrorLog: log.New(
				os.Stdout, "promhttp", log.Ldate|log.Ltime|log.Lshortfile,
			),
			ErrorHandling: promhttp.ContinueOnError,
		})

	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
	})
	log.Println("Start server at :8080")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Println("Error occur when start server %v", err)
		os.Exit(1)
	}

}
