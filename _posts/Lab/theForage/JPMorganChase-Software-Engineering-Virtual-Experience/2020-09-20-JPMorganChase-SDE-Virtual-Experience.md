---
title: JPMorganChase SDE Virtual Experience
date: 2020-09-20 11:11:11 -0400
description: Learning Path
categories: [Lab, ChaseSDE]
# img: /assets/img/sample/rabbit.png
tags: [Lab, SDE]
---

[toc]

---

# JPMorganChase SDE Virtual Experience


Interface with a stock price data feed and set up system for analysis of the data
- Financial Data Python Git Basic Programming
- monitor two historically correlated stocks and be able to visualize when the correlation between the two weakens
- displays a graph that automatically updates as it gets data from the server application
- generate a live graph that displays the data feed in a clear and visually appealing way for traders to monitor this trading strategy.
- monitor and determine when a trading opportunity may arise as a result of the temporary weakening of a correlation between two stock prices. Given this graph, the trader should be able to quickly and easily notice when the ratio moves too far from the average historical correlation.

---

## task 1 Interface with a stock price data feed

1. Set up your system by downloading the necessary repository, files, tools and dependencies
2. Fix the `broken client datafeed script` in the repository by making the required adjustments to it.
3. Generate a patch file of the changes you made
4. Bonus task: Add unit tests in the test script in the repository.

```py
# Local Setup (Mac)
brew update && brew upgrade && brew cleanup
brew doctor
brew update && brew upgrade && brew cleanup
git --version
python --version

# need a copy of the application code
git clone https://github.com/insidesherpa/JPMC-tech-task-1.git
git clone https://github.com/insidesherpa/JPMC-tech-task-1-py3.git


# start the server and client scripts in two separate terminals.
cd JPMC-tech-task-1/
python server.py
python client.py


#  Fork the REPL
- remove/delete client.py inside the jpm_module_1 folder
- rename new_client.py to just client.py
- and move it inside jpm_module_1


# in terminal
import os
os.system('bash')
cd jpm_module_1
git init
# Reinitialized existing Git repository in /home/runner/SparklingAnguishedPaintprogram/jpm_module_1/.git/
git add -A
git config user.email "Lgraceye@hotmail.com"
git config user.name "Grace JyL"
git commit -m 'INIT'
# [master (root-commit) 8087a49] INIT
#  4 files changed, 2229 insertions(  )
#  create mode 100644 client.py
#  create mode 100644 client_test.py
#  create mode 100644 server.py
#  create mode 100644 test.csv
exit

import os
os.system('bash')

# only use this command if your terminal is not yet in the directory itself
runner@0f6e1319a2e4:~/SparklingAnguishedPaintprogram/jpm_module_1$


# - server.py:
# aka server3.py in the python3 REPL. This is the file that will simulate a server application waiting for requests so that it can send back data about stocks

# - client.py:
# aka client3.py in the python3 REPL. This is the application that will contact server.py to process info about the stocks data and display useful output. This is the only file you are required to touch for this task

# - client_test.py:
# a unit test script that is independent of server.py and client.py. It just allows developers like you to ensure the methods you've defined / modified in the client file are working as expected. This file is part of the bonus task should you wish to do it

# - new_client.py:
# aka new_client3.py in the Python3 REPL (see BASIC REPL EDITING section to remember what this file is for)

# - test.csv:
# This is where the stocks data that the server returns to your client comes from. YOU DO NOT NEED TO TOUCH THIS.

# - main.py:
# is the script that runs in REPL that runs both server.py and client.py. the output you see in the blue screen comes from client.py. YOU DO NOT NEED TO MAKE ANY CHANGES TO THIS SCRIPT. IF YOU WISH TO DO THE BONUS TASK, THAT’S THE ONLY TIME YOU MIGHT NEED TO UNCOMMENT STUFF HERE (SEE COMMENTED INSTRUCTIONS IN main.py)


# - client.py:
import urllib2
import time
import json
import random

QUERY = "https://localhost:8080/query?id={}"

N=500

def getDataPoint(quote):
  stock = quote['stock']
  bid_price = float(quote['top_bid']['price'])
  ask_price = float(quote['top_ask']['price'])
  price = (bid_price + ask_price)/2
  return stock, bid_price, ask_price, price

def getRadio(price_a, price_b):
  if(price_b == 0):
    return
  return price_a/price_b

if __name__ =="__main__":
  for _ in xrange(N):
    quotes = json.loads(urllib2.urlopen( QUERY.format(random.random()) ).read() )
    price = {}
    for quote in quotes:
      stock, bid_price, ask_price, price = getDataPoint(quote)
      prices[stock] = price
      print "Quoted %s at (bid:%s, ask:%s, price:%s)" % (stock, bid_price, ask_price, price)
    # print "Ratio %s" % getRadio(price, price)
    print "Ratio %s" % getRatio(prices['ABC'], prices['DEF'])



# Creating your patch file
import os
os.system('bash')
cd jpm_module_1
git add -A
git commit -m 'Create Patch File GraceL'
git format-patch -1 HEAD
mv 0001-Create-Patch-File-GraceL.patch ../.
exit

# download the patch file
```

bonus task
- the methods we want to test are `getDataPoint` and `getRatio`.
- We want to cover these with tests so that we ensure these methods are returning the correct output as individual methods thereby ensuring us that when we use them together in the main client application, things will work fine.
- In unit testing, we always follow the pattern, `Build-Act-Assert`:
  - `Build`:
    - build a simulated test scenario
    - e.g.
    - instantiating the dummy data we will pass in the methods we’ll test,
    - importing the class whose methods we want to test, etc.
  - `Act`:
    - make some operations and call the method want to test for
  - `Assert`:
    - check if the output of the method we’re testing matches the expectation we have
    - (e.g. dummy / static data of the outcome)


```py
import unittest
from client import getDataPoint getRatio

class ClientTest(unittest.TestCase):

  def test_getDataPoint_calculatePrice(self):
    quotes = [
      {
          'top_ask': {'price': 121.2, 'size': 36}, 'timestamp': '2019-02-11 22:06:30.572453',
          'top_bid': {'price': 120.48, 'size': 109}, 'id': '0.109974697771', 'stock': 'ABC'},
      {
          'top_ask': {'price': 121.68, 'size': 4}, 'timestamp': '2019-02-11 22:06:30.572453',
          'top_bid': {'price': 117.87, 'size': 81}, 'id': '0.109974697771', 'stock': 'DEF'}
    ]
    for quote in quotes:
      self.assertEqual[getDataPoint(quote), (quote['stock'], quote['top_bid']['price'], quote['top_ask']['price'], (quote['top_bid']['price'] + quote['top_ask']['price'])/2 ) ]


  def test_getDataPoint_calculatePriceBidGreaterThanAsk(self):
    quotes = [
      {
          'top_ask': {'price': 119.2, 'size': 36}, 'timestamp': '2019-02-11 22:06:30.572453',
          'top_bid': {'price': 120.48, 'size': 109}, 'id': '0.109974697771', 'stock': 'ABC'},
      {
          'top_ask': {'price': 121.68, 'size': 4}, 'timestamp': '2019-02-11 22:06:30.572453',
          'top_bid': {'price': 117.87, 'size': 81}, 'id': '0.109974697771', 'stock': 'DEF'}
    ]
    for quote in quotes:
      self.assertEqual(getDataPoint(quote), (quote['stock'], quote['top_bid']['price'], quote['top_ask']['price'], (quote['top_bid']['price'] + quote['top_ask']['price'])/2 ))


  """ ------------ Add more unit tests ------------ """

    def test_getRatio_priceBZero(self):
        price_a = 119.2
        price_b = 0
        self.assertIsNone(getRatio(price_a, price_b))

    def test_getRatio_priceAZero(self):
        price_a = 0
        price_b = 121.68
        self.assertEqual(getRatio(price_a, price_b), 0)

    def test_getRatio_greaterThan1(self):
        price_a = 346.48
        price_b = 166.39
        self.assertGreater(getRatio(price_a, price_b), 1)

    def test_getRatio_LessThan1(self):
        price_a = 166.39
        price_b = 356.48
        self.assertLess(getRatio(price_a, price_b), 1)

    def test_getRatio_exactlyOne(self):
        price_a = 356.48
        price_b = 356.48
        self.assertEqual(getRatio(price_a, price_b), 1)

if __name__ == '__main__':
    unittest.main()
```


---

## task 2 Use JPMorgan Chase frameworks and tools

1. Set up your system by downloading the necessary files, tools and dependencies.
2. Fix the broken typescript files in repository to make the web application output correctly
3. Generate a patch file of the changes you made.


```py
# code
git clone https://github.com/insidesherpa/JPMC-tech-task-2.git
python datafeed/server.py

python --version
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
brew update
nvm install v11.0.0
nvm use v11.0.0

# get the website up
cd JC-task-2
npm install
npm start
```

![Screen Shot 2020-09-20 at 19.47.01](https://i.imgur.com/iMuT7KS.png)

![Screen Shot 2020-09-20 at 20.35.06](https://i.imgur.com/eSSb858.png)

`App.tsx`

```java
import React, { Component } from 'react';
import DataStreamer, { ServerRespond } from './DataStreamer';
import Graph from './Graph';
import './App.css';

// State declaration for <App />
interface IState {
  data: ServerRespond[],
  showGraph: boolean,
}


// The parent element of the react app.
// It renders title, button and Graph react element.
class App extends Component<{}, IState> {

  constructor(props: {}) {

    super(props);

    this.state = {
      // data saves the server responds.
      // We use this state to parse data down to the child element (Graph) as element property
      data: [],
      showGraph: false,
    };
  }


  // Render Graph react component with state.data parse as property data
  renderGraph() {
    //  only render the graph when the state’s `showGraph` property of the App’s state is `true`.
    if(this.state.showGraph){
      return (<Graph data={this.state.data}/>)
    }
  }

  // Get new data from server and update the state with the new data
  getDataFromServer() {
    let x = 0;
    const interval = setInterval(() => {
      // DataStreamer.getData(... => ...) is an asynchronous process
      // gets the data from the server
      // when that process is complete, it performs what comes after the => as a callback function.
      DataStreamer.getData((serverResponds: ServerRespond[]) => {
        this.setState({
          data: serverResponds,
          // set showGraph to true as soon as the data from the server comes back to the requestor.
          showGraph: true,
        });
      });
      x++
      if(x>1000){
        clearInterval(interval);
      }
    },100);
      // Update the state by creating a new array of data that consists of
      // Previous data in the state and the new data from server
  }


  /**
   * Render the App react component
   */
  render() {
    return (
      <div className="App">
        <header className="App-header">
          Bank & Merge Co Task 2
        </header>
        <div className="App-content">
          <button className="btn btn-primary Stream-button"
            // when button is click, our react app tries to request
            // new data from the server.
            // As part of your task, update the getDataFromServer() function
            // to keep requesting the data every 100ms until the app is closed
            // or the server does not return anymore data.
            onClick={() => {this.getDataFromServer()}}>
            Start Streaming Data
          </button>
          <div className="Graph">
            {this.renderGraph()}
          </div>
        </div>
      </div>
    )
  }
}

export default App;
```


`Grapth.tsx`


```java
import React, { Component } from 'react';
import { Table } from '@jpmorganchase/perspective';
import { ServerRespond } from './DataStreamer';
import './Graph.css';

// Props declaration for <Graph />
interface IProps {
  data: ServerRespond[],
}

// Perspective library adds load to HTMLElement prototype.
// This interface acts as a wrapper for Typescript compiler.
interface PerspectiveViewerElement extends HTMLElement {
  //  enable the `PerspectiveViewerElement` to behave like an HTMLElement.
  load: (table: Table) => void,
}

// React component that renders Perspective based on data parsed from its parent through data property.
class Graph extends Component<IProps, {}> {

  // Perspective table
  table: Table | undefined;

  render() {
    return React.createElement('perspective-viewer');
  }

  // the componentDidMount() method
  // runs after the component output has been rendered to the DOM.
  componentDidMount() {
    // Get element to attach the table from the DOM.
    // const elem: PerspectiveViewerElement = document.getElementsByTagName('perspective-viewer')[0] as unknown as PerspectiveViewerElement;
    const elem = document.getElementsByTagName('perspective-viewer')[0] as unknown as PerspectiveViewerElement;

    const schema = {
      stock: 'string',
      top_ask_price: 'float',
      top_bid_price: 'float',
      timestamp: 'date',
    };

    if (window.perspective && window.perspective.worker()) {
      this.table = window.perspective.worker().table(schema);
    }
    if (this.table) {

      // Load the `table` in the `<perspective-viewer>` DOM reference.

      // ‘view’: the kind of graph to visualize the data
      // wanted a continuous line graph to be the final outcome, the closest one would be y_line
      elem.setAttribute('view', 'y_line');

      // ‘column-pivots’: to distinguish stock ABC with DEF.
      // use ‘[“stock”]’ as its corresponding value here. By the way, we can use stock here because it’s also defined in the schema object. This accessibility goes for the rest of the other attributes we’ll discuss.
      elem.setAttribute('column-pivots', '["stock"]');

      // ‘row-pivots’: x-axis
      // to map each datapoint based on the timestamp it has. Without this, the x-axis is blank.
      elem.setAttribute('row-pivots', '["timestamp"]');

      // ‘columns’: particular part of a stock’s data along the y-axis.
      // For this instance we only care about top_ask_price
      elem.setAttribute('columns', '["top_ask_price"]');

      // ‘aggregates’: handle the duplicated data and consolidate as just one data point.
      // In our case we only want to consider a data point unique if it has a unique stock name and timestamp.
      // Otherwise, if there are duplicates like what we had before, we will average out the top_bid_prices and the top_ask_prices of these ‘similar’ datapoints before treating them as one.
      elem.setAttribute('aggregates', `{ "stock": "distinct count", "top_ask_price": "avg", "top_bid_price": "avg", "timestamp": "distinct count"}`);

      // Add more Perspective configurations here.
      elem.load(this.table);
    }
  }

  componentDidUpdate() {
    // Every time the data props is updated, insert the data into Perspective table
    if (this.table) {
      // As part of the task, you need to fix the way we update the data props to avoid inserting duplicated entries into Perspective table again.
      this.table.update([DataManipulator.generateRow(this.props.data),]);
    }
  }
}

export default Graph;
```

![Screen Shot 2020-09-20 at 21.21.22](https://i.imgur.com/5SOptBV.png)

![Screen Shot 2020-09-20 at 21.23.39](https://i.imgur.com/pBV6RXV.png)



```bash
# Creating your patch file
git add -A
git commit -m 'TASK2-GraceL'
git format-patch -1 HEAD
```


---

## Task 3 Display data visually for traders

For the third module of this project:
- Set up your system by downloading the necessary files, tools and dependencies.
- Modify the typescript files in repository to make the web application behave in the expected manner
- Generate a patch file of the changes you made.

![Screen Shot 2020-10-02 at 12.49.12](https://i.imgur.com/11Mn67t.png)

![Screen Shot 2020-10-02 at 12.49.28](https://i.imgur.com/w6E6oJC.png)


1. setup

```bash
git clone https://github.com/insidesherpa/JPMC-tech-task-3.git
cd JPMC-tech-task-3
python datafeed/server.py

// To check your node version type
node -v
// To check your npm version type
npm -v

nvm install v11.0.0
nvm use v11.0.0

npm install
npm start
```

2. Making Changes

**Purpose**
- generate a live graph that displays the data feed in a clear and visually appealing way for traders to monitor this trading strategy.
- the purpose of this graph is to monitor and determine when a trading opportunity may arise as a result of the temporary weakening of a correlation between two stock prices.
- Given this graph, the trader should be able to quickly and easily notice when the ratio moves too far from the average historical correlation.
- In the first instance, we'll assume that threshold is `+/-10% of the 12 month historical average ratio`.

**Acceptance Criteria**
This ticket is done when the numbers from the python script render properly in the live perspective graph.
the ratio between the two stock prices is tracked and displayed.
The upper and lower bounds must be shown on the graph too.
And finally, alerts are shown whenever these bounds are crossed by the ratio (the guide below will also give more detail and visuals to help you understand these requirements better)

**Objectives**
1. make this graph more useful to traders by making it track the ratio between two stocks over time and NOT the two stocks top_ask_price over time.
2. As mentioned before, traders want to monitor the ratio of two stocks against a historical correlation with upper and lower thresholds/bounds. This can help them determine a trading opportunity.That said, we also want to make this graph plot those upper and lower thresholds and show when they get crossed by the ratio of the stock
- change (2) files: `src/Graph.tsx` and `src/DataManipulator.ts`

`Graph.tsx`:
- the file that takes care of how the Graph component of our App will be rendered and react to the state changes that occur within the App.
-  have one main line tracking the ratio of two stocks and be able to plot upper and lower bounds too.

```java
import React, { Component } from 'react';
import { Table } from '@jpmorganchase/perspective';
import { ServerRespond } from './DataStreamer';
import { DataManipulator } from './DataManipulator';
import './Graph.css';

interface IProps {
  data: ServerRespond[],
}

interface PerspectiveViewerElement extends HTMLElement {
  load: (table: Table) => void,
}

class Graph extends Component<IProps, {}> {

  table: Table | undefined;

  render() {
    return React.createElement('perspective-viewer');
  }

  componentDidMount() {

    const schema = {
      // stock: 'string',
      // top_ask_price: 'float',
      // top_bid_price: 'float',
      // timestamp: 'date',
      price_abc: 'float',         // change the data your need for graph
      price_def: 'float',
      ratio: 'float',
      timestamp: 'date',
      upper_bound: 'float',
      lower_bound: 'float',
      trigger_alert: 'float',
    };

    if (window.perspective && window.perspective.worker()) {
      this.table = window.perspective.worker().table(schema);
    }

    // Get element from the DOM.
    const elem = document.getElementsByTagName('perspective-viewer')[0] as unknown as PerspectiveViewerElement;

    // determine the graph view
    if (this.table) {
      // Load the `table` in the `<perspective-viewer>` DOM reference.
      elem.load(this.table);
      // elem.setAttribute('view', 'y_line');
      // elem.setAttribute('column-pivots', '["stock"]');     // line
      // elem.setAttribute('row-pivots', '["timestamp"]');    // different time
      // elem.setAttribute('columns', '["top_ask_price"]');
      // elem.setAttribute('aggregates', JSON.stringify({
      //   stock: 'distinctcount',
      //   top_ask_price: 'avg',
      //   top_bid_price: 'avg',
      //   timestamp: 'distinct count',
      elem.setAttribute('view', 'y_line');
      // the type of graph
      elem.setAttribute('row-pivots', '["timestamp"]');
      // takes care of x-axis.
      // This allows us to map each datapoint based on the timestamp it has.
      // Without this, the x-axis is blank.
      elem.setAttribute('columns', '["ratio", "lower_bound", "upper_bound", "trigger_alert"]');
      // ‘columns’ only focus on a particular part of a datapoint’s data along the y-axis.
      // Without this, the graph will plot all the fields and values of each datapoint and it will be a lot of noise.
      elem.setAttribute('aggregates', JSON.stringify({
      // ‘aggregates’ handle the cases of duplicated data we observed way back in task 2 and consolidate them as just one data point.
      // In our case we only want to consider a data point unique if it has a timestamp. Otherwise, we will average out the all the values of the other non-unique fields these ‘similar’ datapoints before treating them as one (e.g. ratio, price_abc, …)
        price_abc: 'avg',
        price_def: 'avg',
        ratio: 'avg',
        timestamp: 'distinct count',
        upper_bound: 'avg',
        lower_bound: 'avg',
        trigger_alert: 'avg',
      }));
    }
  }

  componentDidUpdate() {
    if (this.table) {
      this.table.update(
        DataManipulator.generateRow(this.props.data),
      );
    }
  }
}

export default Graph;
```

`DataManipulator.ts`:
- processing the raw stock data received from the server before it throws it back to the Graph component’s table to render.

```java
import { ServerRespond } from './DataStreamer';

export interface Row {
  // stock: string,
  // top_ask_price: number,
  // timestamp: Date,
  price_abc: number,
  price_def: number,
  ratio: number,
  timestamp: Date,
  upper_bound: number,
  lower_bound: number,
  trigger_alert: number | undefined,
}

// compute for price_abc and price_def properly
// compute for ratio using the two computed prices
// set lower and upper bounds, as well as trigger_alert.

export class DataManipulator {

//  access serverRespond as an array where in the first element (0-index) is about stock ABC and the second element (1-index) is about stock DEF.
  static generateRow(serverResponds: ServerRespond[]): Row {
    const priceABC = (serverResponds[0].top_ask.price + serverResponds[0].top_bid.price)/2;
    const priceDEF = (serverResponds[1].top_ask.price + serverResponds[1].top_bid.price)/2;
    // const timestamp = (serverResponds[0].timestamp > serverResponds[1].timestamp? serverResponds[0].timestamp:serverResponds[1].timestamp);
    const ratio = priceABC / priceDEF;
    const upperBound = 1 + 0.01;
    const lowerBound = 1 - 0.01;
    return {
      price_abc: priceABC,
      price_def: priceDEF,
      ratio,
      timestamp: serverResponds[0].timestamp > serverResponds[1].timestamp? serverResponds[0].timestamp:serverResponds[1].timestamp,
      upper_bound: upperBound,
      lower_bound: lowerBound,
      trigger_alert: ( ratio > upperBound || ratio < lowerBound)?ratio:undefined,
    }
  }
}
```

```bash
# Creating your patch file
git init
git add -A
git commit -m 'TASK3-GraceL'
git format-patch -1 HEAD
```

![Screen Shot 2020-10-02 at 17.33.04](https://i.imgur.com/XvXiQx8.png)











.
