---
title: Kubernetes ReplicaSet Controller Design Principle
author: Hulua
date: 2022-04-27 20:55:00 +0800
categories: [Kubernetes, Source Code]
tags: [kubernetes, debug, source code]
---

In kubernetes, controllers play a vital role to orchestrate resources. For beginners learning kubernetes, we need to understand what are controllers. In short, controllers manipulate resources. In particular, resources indicate a collection of static object definition, and they exist in the ETCD database (managed by ApiServer). For a simple example, you can throw the below pod yaml file
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
    ports:
    - containerPort: 80
```
into ETCD using kubectl, then you have a Pod resource. However, before scheduler arranges it to run in some node and kubelet starts the runtime container, the resource is just a static definition. This sounds like an order for a meal, it is just there and may not be cooked/served yet. The role of controllers is to orchestrate a lot of orders (but controllers do not cook or serve dishes). As another example, say if you throw a ReplicaSet with 3 replicas into ETCD, replicatset controller will then create 3 corresponding Pod resources in ETCD. Again they just make the order, the actual work to launch containers is the responsibility of kubelet. As such, you can immagine that controller development/debug does not require container runtime at all.

ReplicaSet controller in k8s is a very basic controller. In this post, we will have a look at the implementation details for ReplicatSet controller.

# Run ReplicatSet Controller
Earlier I shared a post on how to debug k8s source code. However, that solution is too cubersome. In fact, we have a lighter way if we just want to run/debug simple function unit. Let's go the the k8s source code folder and run:
```console
kubernetes git:(master) go test -v ./pkg/controller/replicaset -run TestWatchPods
=== RUN   TestWatchPods
I0427 10:34:13.498307   65353 replica_set.go:205] Starting replicaset controller
I0427 10:34:13.498605   65353 shared_informer.go:255] Waiting for caches to sync for ReplicaSet
I0427 10:34:13.498619   65353 shared_informer.go:262] Caches are synced for ReplicaSet
--- PASS: TestWatchPods (0.00s)
PASS
ok  	k8s.io/kubernetes/pkg/controller/replicaset	0.789s
```
Now, you see we started a replicaset controller and did a test. The details of this test can be found `./pkg/controller/replicaset/replicat_set_test.go`. You can also do a step-by-step debug using:
```console
➜  kubernetes git:(master) dlv test ./pkg/controller/replicaset/

Type 'help' for list of commands.
(dlv) b TestWatchPods
Breakpoint 1 set at 0x2afd2d2 for k8s.io/kubernetes/pkg/controller/replicaset.TestWatchPods() ./pkg/controller/replicaset/replica_set_test.go:708
(dlv) c
=> 708:	func TestWatchPods(t *testing.T) {
   709:		client := fake.NewSimpleClientset()
   710:
   711:		fakeWatch := watch.NewFake()
   712:		client.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))
```
Then you will be able to go through the steps of creating the controller and see how it works.

# ReplicaSet Controller Design Principle

The ultimate goal of replicat set controller is to ensure there are always a dedicated number of pods (here we can only ensure from the resource manifest level, but we cannot guarantee cooresponding containers can be created). Before we have look at the implementation, it is helpful if we ask the following questions:

 *How to deal with the case when controllers get restarted?*
 
Cause the controllers may be interrupted but the number of pods is not enough. The overview is count the existing number of pods *at the begining of every sync* loop. Each managed pod is expected to have the same labels and may have a owner reference points to the replicat set. Then depending on the number, we keep creating or deleting pods to statisfy the dedidated number.

# Implementation Details

We will take a look at a few key implementation details. The ReplicatSetController is defined as a struct:

```go
type ReplicaSetController struct {
	schema.GroupVersionKind
	kubeClient clientset.Interface
	
    //This is podController interface, which will be responsible to create or delete pod 
    //resources.
	podControl controller.PodControlInterface
	
	burstReplicas int
	syncHandler func(ctx context.Context, rsKey string) error

	// Think about what if we want to create 200 Pods in one sync, but before it finishes
	// a second sync is started? Here in the first sync, expecations will say I expect to 
	// create 200 pods. Thus if the expections are not satisfied, second sync  will quit
	expectations *controller.UIDTrackingControllerExpectations


	rsLister appslisters.ReplicaSetLister
	rsListerSynced cache.InformerSynced
	
    //This is storage for objects
	rsIndexer      cache.Indexer


	podLister corelisters.PodLister
	
	podListerSynced cache.InformerSynced

	// This is work queue. items in this queue will be processed one by one
	queue workqueue.RateLimitingInterface
}

```

When the ReplicatSet controller is created, it will listen to Replicat Set and Pods Add/Update/Delete Events.

```go
func NewBaseController(rsInformer appsinformers.ReplicaSetInformer, podInformer coreinformers.PodInformer, kubeClient clientset.Interface, burstReplicas int,
	gvk schema.GroupVersionKind, metricOwnerName, queueName string, podControl controller.PodControlInterface) *ReplicaSetController {
	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage(metricOwnerName, kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

    ...
	rsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rsc.addRS,
		UpdateFunc: rsc.updateRS,
		DeleteFunc: rsc.deleteRS,
	})
	...
		podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: rsc.addPod,
		// This invokes the ReplicaSet for every pod change, eg: host assignment. Though this might seem like
		// overkill the most frequent pod update is status, and the associated ReplicaSet will only list from
		// local storage, so it should be ok.
		UpdateFunc: rsc.updatePod,
		DeleteFunc: rsc.deletePod,
	})
```

For which ever detected, the corresponding action is to add the respective ReplicatSet resource into the work queue. For instance:

```go
func (rsc *ReplicaSetController) addRS(obj interface{}) {
	rs := obj.(*apps.ReplicaSet)
	klog.V(4).Infof("Adding %s %s/%s", rsc.Kind, rs.Namespace, rs.Name)
	rsc.enqueueRS(rs)
}
func (rsc *ReplicaSetController) enqueueRS(rs *apps.ReplicaSet) {
	key, err := controller.KeyFunc(rs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %#v: %v", rs, err))
		return
	}
	rsc.queue.Add(key)
}
```

Now we can have a look at the control loop. For each worker process, it will take one item from the work queue and call `syncHandler` to start a sync loop.

```go
func (rsc *ReplicaSetController) processNextWorkItem(ctx context.Context) bool {
	key, quit := rsc.queue.Get()
	if quit {
		return false
	}
	defer rsc.queue.Done(key)

	err := rsc.syncHandler(ctx, key.(string))
	if err == nil {
		rsc.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("sync %q failed with %v", key, err))
	rsc.queue.AddRateLimited(key)

	return true
}
```

Here `syncHandler` points to `syncReplicaSet`:

```go
//Sync a specific replicat set, key is the fullname for the replicatset
func (rsc *ReplicaSetController) syncReplicaSet(ctx context.Context, key string) error {
	...
    // Get namespace and anme
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
    
    // Get the replicatset object
	rs, err := rsc.rsLister.ReplicaSets(namespace).Get(name)

    // Check if the last round of sync is satisfied (finished)
	rsNeedsSync := rsc.expectations.SatisfiedExpectations(key)
	
    // First get all the pods in that namespace
	allPods, err := rsc.podLister.Pods(rs.Namespace).List(labels.Everything())


	// Ignore inactive pods.
	filteredPods := controller.FilterActivePods(allPods)


    // Now get all existing pods for the replicatset
	filteredPods, err = rsc.claimPods(ctx, rs, selector, filteredPods)

	var manageReplicasErr error
	if rsNeedsSync && rs.DeletionTimestamp == nil {
	
        // Now we depends on how many existing pods
        // either create new pods or delete pods
		manageReplicasErr = rsc.manageReplicas(ctx, filteredPods, rs)
	}
	....
```

The actual work to create new pods or delete pods will be done in `manageReplicas`.

```go
func (rsc *ReplicaSetController) manageReplicas(ctx context.Context, filteredPods []*v1.Pod, rs *apps.ReplicaSet) error {

    //Caclulate the difference, either to create a delete
	diff := len(filteredPods) - int(*(rs.Spec.Replicas))
	rsKey, err := controller.KeyFunc(rs)
	...
	if diff < 0 {
		diff *= -1
		if diff > rsc.burstReplicas {
			diff = rsc.burstReplicas
		}
		// Set the number of pods expected to create, for each pod 
		// that is created succefully, this number will decrease 1.
		// When pod add event is detected, there is a call to 
		// rsc.expectations.CreationObserved(rsKey) to reduce this number
		// The next round of sync will quit if this number is not 0.
		rsc.expectations.ExpectCreations(rsKey, diff)
		klog.V(2).InfoS("Too few replicas", "replicaSet", klog.KObj(rs), "need", *(rs.Spec.Replicas), "creating", diff)
		

		successfulCreations, err := slowStartBatch(diff, controller.SlowStartInitialBatchSize, func() error {
		    // Actual create pods work is delegated to podControl
			err := rsc.podControl.CreatePods(ctx, rs.Namespace, &rs.Spec.Template, rs, metav1.NewControllerRef(rs, rsc.GroupVersionKind))
			if err != nil {
				if apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
					return nil
				}
			}
			return err
		})
        ....
	}
```

Actual pods creation work is wrapped in `slowStartBatch`, there pods will be created batch by batch with different batch sizes. 
```go
        errCh := make(chan error, batchSize)
		var wg sync.WaitGroup
		wg.Add(batchSize)
		for i := 0; i < batchSize; i++ {
			go func() {
				defer wg.Done()
				//Here fn is the function passed from the parameter of slowStartBatch
				if err := fn(); err != nil {
					errCh <- err
				}
			}()
		}
		wg.Wait()
```

# Summary

In this post, we had an overview of the principle of replicat set controller in k8s. At each sync loop, the controller will claim pods (i.e., count how many pods exists under the replicat set), then decide whether to create more pods or delete pods. There are some hightlight implementation tricks to pay attention, such as use expectations to avoid concurrent sync loop, and use slow start batch to make pod creation process smooth.