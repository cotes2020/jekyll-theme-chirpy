package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// I wrap three Prometheus metrics’ into a struct.
type PrometheusHttpMetric struct {
	Prefix                string
	ClientConnected       prometheus.Gauge
	TransactionTotal      *prometheus.CounterVec
	ResponseTimeHistogram *prometheus.HistogramVec
	Buckets               []float64
}

// initialize metrics with a prefix (or called as a namespace)
// so that the metrics won’t conflict with others.
// The TransactionTotal is created as a vector so that different labels such as handler and HTTP status can be segregated.
// The response_time requires a float array for the predefined buckets.
func InitPrometheusHttpMetric(prefix string, buckets []float64) *PrometheusHttpMetric {
	phm := PrometheusHttpMetric{
		Prefix: prefix,
		ClientConnected: promauto.NewGauge(prometheus.GaugeOpts{
			Name: prefix + "_client_connected",
			Help: "Number of active client connections",
		}),
		TransactionTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: prefix + "_requests_total",
			Help: "total HTTP requests processed",
		}, []string{"code", "method"},
		),
		ResponseTimeHistogram: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    prefix + "_response_time",
			Help:    "Histogram of response time for handler",
			Buckets: buckets,
		}, []string{"handler", "method"}),
	}
	return &phm
}

// Prometheus client API provides the HTTP middleware.
// wrap the HTTP handler with the Prometheus different type of middlewares.
// takes a type of http.HandlerFunc, wraps the three custom metrics implementer
// and return a http.handler for setting the route.
func (phm *PrometheusHttpMetric) WrapHandler(handlerLabel string, handlerFunc http.HandlerFunc) http.Handler {
	handle := http.HandlerFunc(handlerFunc)
	wrappedHandler := promhttp.InstrumentHandlerInFlight(
		phm.ClientConnected,
		promhttp.InstrumentHandlerCounter(
			phm.TransactionTotal,
			promhttp.InstrumentHandlerDuration(
				phm.ResponseTimeHistogram.MustCurryWith(
					prometheus.Labels{"handler": handlerLabel}),
				handle),
		),
	)
	return wrappedHandler
}

// The main handler of the sample app just parses the URL query parameter
// and sleeps enough to simulate the latency of the service.
func myHandler(w http.ResponseWriter, r *http.Request) {
	cost := r.FormValue("cost")
	val, err := strconv.ParseFloat(cost, 64)
	if err != nil {
		http.Error(w, "Fail to convert cost as float value", 500)
		return
	}
	sleep := time.Duration(val*1e+9) * time.Nanosecond
	time.Sleep(sleep)
	fmt.Fprintf(w, "Time spend for this request: %.2f", sleep.Seconds())
}

func main() {
	phm := InitPrometheusHttpMetric(
		"myapp",
		prometheus.LinearBuckets(0, 5, 20))

	http.Handle("/metrics", promhttp.Handler())
	http.Handle("/service", phm.WrapHandler("myhandler", myHandler))

	port := os.Getenv("LISTENING_PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("listening on port:%s", port)

	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatalf("Failed to start server:%v", err)
	}
}
