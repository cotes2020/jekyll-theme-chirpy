package myapp_prometheus

import (
	"log"
	"net/http"
	"os"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// =========== myapp Prometheus Step up ===========

// SetupPrometheus
// set up configuration for Prometheus
var (
	myapp_Prometheus_Port = ":9999"
	myapp_Prometheus_Path = "/Mymetrics"
)

func SetupPrometheus() []string {
	log.Println("+++ SetupPrometheus:")
	p_port := os.Getenv("MP_PORT")
	if p_port == "" {
		p_port = myapp_Prometheus_Port
		// p_port = ":8080"
	}
	p_path := os.Getenv("MP_PATH")
	if p_path == "" {
		p_path = myapp_Prometheus_Path
		// p_path = "/metrics"
	}
	config := []string{p_port, p_path}
	log.Printf(
		"+++ Prometheus Config: path %s, port %s",
		config[1], config[0])
	return config
}

// SetupHttpServerwithRegistry
// set up HttpServer with non-global registry
func SetupHttpServerwithRegistry() string {
	// func SetupHttpServerwithRegistry() (string, *prometheus.Registry) {

	config := SetupPrometheus()
	p_port := config[0]
	p_path := config[1]

	// // Create non-global registry.
	// log.Println("+++ Create Registry:")
	// reg := prometheus.NewRegistry()

	// gatherers := prometheus.Gatherers{
	// 	// prometheus.DefaultGatherer,
	// 	reg,
	// }

	// handler := promhttp.HandlerFor(
	// 	gatherers,
	// 	promhttp.HandlerOpts{
	// 		ErrorLog: log.New(
	// 			os.Stdout, "promhttp", log.Ldate|log.Ltime|log.Lshortfile,
	// 		),
	// 		ErrorHandling: promhttp.ContinueOnError,
	// 		// Pass custom registry
	// 		Registry: reg,
	// 	})

	// // http.Handle(p_path, handler)
	// http.HandleFunc(p_path, func(w http.ResponseWriter, r *http.Request) {
	// 	handler.ServeHTTP(w, r)
	// })

	// go func() {
	// 	log.Printf("+++ Listening on server:")
	// 	// log.Fatal(http.ListenAndServe(config[0], nil))
	// 	if err := http.ListenAndServe(p_port, nil); err != nil {
	// 		log.Println("Error occur when start server %v", err)
	// 		os.Exit(1)
	// 	}
	// 	// To test:
	// 	// curl -H 'Accept: application/openmetrics-text' localhost:8082/metrics
	// }()

	http.Handle(p_path, promhttp.Handler())

	// err := http.ListenAndServe(p_port, nil)
	// log.Fatal(err)

	go func() {
		log.Printf("+++ Listening on server:")
		// log.Fatal(http.ListenAndServe(config[0], nil))
		if err := http.ListenAndServe(p_port, nil); err != nil {
			log.Println("Error occur when start server %v", err)
			os.Exit(1)
		}
		// To test:
		// curl -H 'Accept: application/openmetrics-text' localhost:8082/metrics
	}()

	// return p_port, reg
	return p_port

}

// MetricRegistry
// Registry the metric object
func MetricRegistry(reg *prometheus.Registry) {

	log.Println("+++ RegisterMetric:")
	myapp_core_01 := NewClusterManager("myapp_core_01")
	reg.MustRegister(myapp_core_01)

	metric_test := NewmyappCoreMetrics("myapp_core_test")
	reg.MustRegister(metric_test)

}

// func main() {
// 	p_port, reg := SetupHttpServerwithRegistry()

// 	MetricRegistry(reg)

// 	log.Printf("+++ Listening on server:")
// 	if err := http.ListenAndServe(p_port, nil); err != nil {
// 		log.Println("Error occur when start server %v", err)
// 		os.Exit(1)
// 	}

// }
