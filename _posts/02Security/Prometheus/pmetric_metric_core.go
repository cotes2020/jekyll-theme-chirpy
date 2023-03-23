package medusa_prometheus

import (
	"log"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	dto "github.com/prometheus/client_model/go"
)

var (
	App_Func_Evaluate_Time = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "App_func_evaluate_time",
			Help: "..",
		},
	)
	App_Errors_Total_Count = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "App_errors_total_count",
			Help: "..",
		},
		[]string{"label"},
	)
)

var gaugeMetricList = map[string]prometheus.Gauge{
	"App_Func_Evaluate_Time": App_Func_Evaluate_Time,
}
var countMetricList = map[string]*prometheus.CounterVec{
	"App_Errors_Total_Count": App_Errors_Total_Count,
}

func SetRunTime(start time.Time, funcname string) {
	var elapsed = time.Since(start)
	proMetric := gaugeMetricList[funcname]
	proMetric.Set(float64(elapsed))
	// log.Printf("%s - %s", funcname, elapsed)
}

func GetCounterValue(label string, metric *prometheus.CounterVec) float64 {
	var f = &dto.Metric{}
	err := metric.WithLabelValues(label).Write(f)
	if err != nil {
		log.Println("err")
		return 0
	}
	return f.Counter.GetValue()
}

func IncCount(funcname string) {
	proMetric := countMetricList[funcname]
	var label = "label"
	proMetric.WithLabelValues(label).Inc()
	value := GetCounterValue(label, proMetric)
	log.Printf("---%s - %s", funcname, value)
}

IncCount("App_Errors_Total_Count")
defer SetRunTime(time.Now(), "App_Func_Evaluate_Time")
