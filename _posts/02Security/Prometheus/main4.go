package prometheus

import (
	"flag"
	"log"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	addr              = flag.String("listen-address", ":8080", "The address to listen on for HTTP requests.")
	uniformDomain     = flag.Float64("uniform.domain", 0.0002, "The domain for the a distribution.")
	// normDomain        = flag.Float64("normal.domain", 0.0002, "The domain for the b distribution.")
	// normMean          = flag.Float64("normal.mean", 0.00001, "The mean for the c distribution.")
	oscillationPeriod = flag.Duration("oscillation-period", 10*time.Minute, "The duration of the rate oscillation period.")
)

var (
	// 1. 创建 Prometheus 数据Metric, 就相当于SQL 数据库 声明table
	// Create a summary to track fictional interservice RPC latencies
	// for three distinct services with different latency distributions.
	// These services are differentiated via a "service" label.
	RpcDurations = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name:       "rpc_durations_seconds",
			Help:       "RPC latency distributions.这个metric的帮助信息,metric的项目作用说明",
			Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
		},
		[]string{"service"},
	)
	// The same as above, but now as a histogram, and only for the normal distribution.
	// The buckets are targeted to the parameters of the normal distribution,
	// with 20 buckets centered on the mean, each half-sigma wide.
	RpcDurationsHistogram = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "rpc_durations_histogram_seconds",
		Help:    "RPC latency distributions.这个metric的帮助信息,metric的项目作用说明",
		Buckets: prometheus.LinearBuckets(*normMean-5**normDomain, .5**normDomain, 20),
	})
)

func init() {
	// Register the summary and the histogram with Prometheus's default registry.
	prometheus.MustRegister(RpcDurations)
	prometheus.MustRegister(RpcDurationsHistogram)
	// Add Go module build info.
	prometheus.MustRegister(prometheus.NewBuildInfoCollector())
}

func main() {
	flag.Parse()

	start := time.Now()

	oscillationFactor := func() float64 {
		return 2 + math.Sin(math.Sin(
			2*math.Pi*float64(time.Since(start)) / float64(*oscillationPeriod))
		)
	}

	// 3. 业务在无代码中想插入对时序书库TSDB数据想的数据写入操作,相当与SQL insert
	// Periodically record some sample latencies for the three services.
	go func() {
		for {
			v := rand.Float64() * *uniformDomain
			RpcDurations.WithLabelValues("uniform").Observe(v)
			time.Sleep(time.Duration(100*oscillationFactor()) * time.Millisecond)
		}
	}()

	go func() {
		for {
			v := (rand.NormFloat64() * *normDomain) + *normMean
			RpcDurations.WithLabelValues("normal").Observe(v)
			RpcDurationsHistogram.Observe(v)
			time.Sleep(time.Duration(75*oscillationFactor()) * time.Millisecond)
		}
	}()

	go func() {
		for {
			v := rand.ExpFloat64() / 1e6
			RpcDurations.WithLabelValues("exponential").Observe(v)
			time.Sleep(time.Duration(50*oscillationFactor()) * time.Millisecond)
		}
	}()

	// 4. 提供HTTP API接口,让Prometheus 主程序定时来收集时序数据
	// Expose the registered metrics via HTTP.
	http.Handle("/metrics", promhttp.Handler())
	log.Fatal(http.ListenAndServe(*addr, nil))
}
