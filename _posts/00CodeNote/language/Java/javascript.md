- [Java Script](#java-script)
  - [example](#example)
  - [assert](#assert)


# Java Script

## example

```js
var express = require('express');
var app = express();

app.get('/', function (req, res) {
  res.send('Hello! World!');
});

var server = app.listen(3000, function () {
  var host = server.address().address;
  var port = server.address().port;
  console.log('Example app listening at https://%s:%s', host, port);
});
```



---

## assert


```js
// wtest.js
var assert = require("assert")

describe('Array', function(){
  describe(' #indexOf()', function(){
    it('should return -1 when the value is not present', function(){
      assert.equal(-1, [1,2,3].indexOf(0));
    })
  })
})
```

---
