---
layout: post
title: DSAGV Guide
date: '2024-05-19 00:26:55 +0330'
---

In this post, we cover all essential cpp files of DSAGV app, and then continue to explore ways to re-implement them in a state-of-the-art fashion!

## The structure of <code>PortApp1_0_10.cpp</code>
The `PortApp1_0_10.cpp` file uses the following __forms__ and __translation units__:
```c++
USERES("PortApp1_0_10.res");
------------------------------------------------
USEFORM("Main1_0_5.cpp", MainForm);
USEFORM("MCFModel1_3.cpp", MCFAlgorithmForm);
USEFORM("OpenPort.cpp", OpenPortForm);
USEFORM("PortAbout.cpp", AboutForm);
USEFORM("PortAGV.cpp", PortAGVForm);
USEFORM("PortContainer.cpp", PortContainerForm);
USEFORM("PortLayout.cpp", PortLayoutForm);
USEFORM("Splash.cpp", SplashForm);
USEFORM("PortBenchmark.cpp", BenchOptionForm);
------------------------------------------------
USEUNIT("PBEAMPP4.cpp");
USEUNIT("PBEAMPP1.cpp");
USEUNIT("PBEAMPP2.cpp");
USEUNIT("PBEAMPP3.cpp");
USEUNIT("Dijkstra.cpp");
USEUNIT("PFLOWUP.cpp");
USEUNIT("TREEUP.cpp");
USEUNIT("READMIN.Cpp");
USEUNIT("PREPAIR.cpp");
USEUNIT("OUTPUT.cpp");
USEUNIT("PSIMPLEX1_3.cpp");
USEUNIT("Mcfutil.cpp");
USEUNIT("PBLA1_3.cpp");
USEUNIT("MCFLIGHT1_0_6.Cpp");
USEUNIT("PSTART.cpp");
```
- Note that the keyword `extern` (explained [here](https://learn.microsoft.com/en-us/cpp/cpp/program-and-linkage-cpp?view=msvc-170)) is used for _private global variables (or external linkage)._ It is not necessary for __free functions__ and __non-const variables__. The `extern` keyword is used in `"MCFModel1_3.cpp"` to reference `MCF_NSA_Solve(...)` method defined in `"MCFLIGHT1_0_6.Cpp"`.

## Examining Essential Parts of the `MCFModel1_3.cpp`

This code is really enormous, containing more 4000 line!! We'll go through different parts of it.
