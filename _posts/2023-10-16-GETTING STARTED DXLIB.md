---
title: Getting Started with dxlib
date: 2023-10-16 14:14:00 +/-TTTT
categories:
  - Tutorials
---
_Last updated: 2023-10-16_ ⌛

Hello!
In this post I'll be giving a quick guide to installing, setting up and using the __dxlib__ library in Python!

Be sure to use Python >= 3. To install, simply use the PyPi repository:
```bash
pip install dxlib
```

After installing, you'll have access to some very useful modules, such as:

```
dxlib/
├─ Core/
├─ Strategies/
├─ Managers/
├─ API/
```


## Modules

### Core

The core module has different basic structures for dealing with portfolios, data instances, strategies and structured securities, prices, etc.

The most important structures are:

* __History__: The history class takes care of centralizing technical, numerical and qualitative indicators, such as statistical value calculations, storing and transforming price series, and much more.
* __Portfolio__: The portfolio encapsulates multiple histories and securities, allowing privately managed operations, such as local trading, operation costs calculations, and inventory management.
* __Security__: A security is the internally representation of real securities, such as future, forward or options contracts, or the more commonly used equities.

All core structures are available directly through the default library import clausule:

```
from dxlib import History, Portfolio, Security, ...
```
