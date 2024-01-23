package main

import (
	"log"
	"math/rand"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// // =========== Custom Metric ===========
// var (
// 	RpcDurations        = My_APPprometheus.RpcDurations
// 	cpuTemp             = My_APPprometheus.CpuTemp
// 	hdFailures          = My_APPprometheus.HdFailures
// 	get_query_url_error = My_APPprometheus.GET_QUEUE_URL_ERR
// 	get_client_error    = My_APPprometheus.GET_CLIENTS_ERR
// )

// =========== Exporter ===========
type ClusterManager struct {
	ClusterName  string
	OOMCountDesc *prometheus.Desc
	RAMUsageDesc *prometheus.Desc
}

// =========== Custom Metric ===========
func NewClusterManager(clusterName string) *ClusterManager {
	return &ClusterManager{
		ClusterName: clusterName,
		OOMCountDesc: prometheus.NewDesc(
			"a_clustermanager_oom_crashes_total",
			"Number of OOM crashes.",
			[]string{"host"}, prometheus.Labels{"clusterName": clusterName},
		),
		RAMUsageDesc: prometheus.NewDesc(
			"a_clustermanager_ram_usage_bytes",
			"RAM usage as reported to the cluster manager.",
			[]string{"host"}, prometheus.Labels{"clusterName": clusterName},
		),
	}
}

// =========== Collector ===========
func (c *ClusterManager) receiveMetric() (
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
	oomCountByHost, ramUsageByHost := c.receiveMetric()
	for host, oomCount := range oomCountByHost {
		ch <- prometheus.MustNewConstMetric(
			c.OOMCountDesc,
			prometheus.CounterValue,
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

// =========== My_APP Prometheus Step up ===========
var (
	My_APP_Prometheus_Port = ":8080"
	My_APP_Prometheus_Path = "/metrics"
)

func SetupPrometheus() []string {
	log.Println("+++ SetupPrometheus:")
	p_port := os.Getenv("MP_PORT")
	if p_port == "" {
		p_port = My_APP_Prometheus_Port
		// p_port = ":8080"
	}
	p_path := os.Getenv("MP_PATH")
	if p_path == "" {
		p_path = My_APP_Prometheus_Path
		// p_path = "/metrics"
	}
	config := []string{p_port, p_path}
	log.Printf(
		"+++ Prometheus Config: path %s, port %s",
		config[1], config[0])
	return config
}

func SetupMetricCollector() http.Handler {

	My_APP_core_01 := NewClusterManager("My_APP_core_01")

	// Create non-global registry.
	reg := prometheus.NewRegistry()

	// RegisterMetric
	log.Println("+++ RegisterMetric: xxxxxxxxx")
	reg.MustRegister(My_APP_core_01)

	gatherers := prometheus.Gatherers{
		// prometheus.DefaultGatherer,
		reg,
	}

	handler := promhttp.HandlerFor(
		gatherers,
		promhttp.HandlerOpts{
			ErrorLog: log.New(
				os.Stdout, "promhttp", log.Ldate|log.Ltime|log.Lshortfile,
			),
			ErrorHandling: promhttp.ContinueOnError,
			// Pass custom registry
			Registry: reg,
		})
	return handler
}

func main() {
	handler := SetupMetricCollector()
	config := SetupPrometheus()
	p_port, p_path := config[0], config[1]

	http.HandleFunc(p_path, func(w http.ResponseWriter, r *http.Request) {
		handler.ServeHTTP(w, r)
	})

	log.Printf("+++ Listening on server:")

	if err := http.ListenAndServe(p_port, nil); err != nil {
		log.Println("Error occur when start server %v", err)
		os.Exit(1)
	}

}
