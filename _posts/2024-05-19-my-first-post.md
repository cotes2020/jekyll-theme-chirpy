---
authors: amireza007
layout: post
title: DSAGV Guide
date: '2024-05-19 00:26:55 +0330'
categories: [DSAGV]
tags: [Scheduling Algorithms, Vehicle Routing Problem, Minimum Cost Flow, Network Simplex, Automated Guided Vehicles, Integer Programming, AGVs]
---

In this post, we'll examine all important cpp files of DSAGV app, and then continue to explore ways to re-implement them in a state-of-the-art fashion!

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
  
---
## <code>mcfdefs.h</code>, <code>mcf.h</code>
- The `mcfdefs.h` contains the **fields** (parameters and nodes) of the `MCFModel` and `mcf.h` contains **methods**. 
Examples of the components of `mcfdefs.h`: (A description of Nodes can be found [here](https://github.com/amireza007/amireza007.github.io/blob/master/assets/mcfdefs.pdf))
```cpp 
struct MCF_node{ ///MCF_node_p is the alias (or typeof) of MCF_node
    int indent;
    int pre_indent;
    MCF_node_p child;
    MCF_node_p right_sibling;     
    MCF_node_p left_sibling;     
    ....
}
```

- Examples of the components of `mcf.h`: 
```cpp
extern long MCF_write_intermediate2(MCF_network_p net);
extern long MCF_primal_start_artificial (MCF_network_p net);
extern long MCF_primal_net_simplex (MCF_network_p net)
```
**Please note that the comments ("_documentation_") are actually good and pretty important in these two cpp files!**

 ## <code>PBLA1_3.cpp</code>, <code>PXIMPLEX1_3.cpp</code>, <code>TREEUPS.cpp</code>, <code>MCFLIGHT1_0_6.cpp</code>
- <a href="#pbla1_3cppp-and-psimplexcpp">`PBLA`</a> and <a href="#pbla1_3cppp-and-psimplexcpp">`PSIMPLEX1_3`</a> and <a href="#mcfutilcpp-mcflight1_0_6cpp-and-mcfmodel1_3cpp">`MCFLIGHT1_0_6`</a> are closely related! There are some useful comments in them, none of which I understand as I still haven't reviewed the mcf problem and NSA algorithm thoroughly.

## <code>PBLA1_3.cppp</code> and <code>PSIMPLEX.cpp</code>
- The `PBLA` (which most likely stands for "problem of best leaving arc") only contains the _mysterious_:
```cpp
MCF_node_p MCF_primal_iminus(MCF_flow_p delta,
                             long       *xchange,
                             MCF_node_p iplus,
                             MCF_node_p jplus,
                             MCF_node_p *w     )
```
- The `PSIMPLEX` (The bigger cpp file with `"mcfdefs.h`",`"pbeadef.h"`,`"pbea.h"`,`"pbla.h"`,`"pflowup.h"`,`"treeup.h"`,`"mcfutil.h"`) only contains the _mysterious_:
```cpp
long MCF_primal_net_simplex(MCF_network_p net)
```
- `TREEUP` is used in `PSIMPLEX1_3`.

## <code>Mcfutil.cpp</code>, <code>MCFLIGHT1_0_6.cpp</code>, and <code>MCFModel1_3.cpp</code>
- portappdatabase is updated within MCFModel1_3.cpp.
- `MCFModel1_3` uses a header file, called **`Global.h`**, which contains strtuctures `Port_Buff`, `Container_Buff[Maximum_Container_Jobs]`, `Crane_Buff[Maximum_Number_Cranes]`, `Job_Crane_Buff[Maximum_Number_Cranes]`, `Vehicle_Buff[Maximum_Number_Vehicles]`, `Route_Buff[Maximum_Number_Junctions];`, `Route_Buff2[Maximum_Number_Lanes]`
- The part that uses <a href="#the-job-generator-hcdvrpcpp">`HCDVRP` </a>
```cpp
void TMCFAlgorithmForm::Handle_Multi_Load_AGVS();
```

## The Job Generator <code>HCDVRP.cpp</code>

`HCDVRP` consists of:
- AGV simulation (such as its properties, its actions) + Container simulation.

```  cpp
struct Container:
{
    int        JobNo      ;
    int        Demand     ;
    int        AppointmentTime;  // ReadyTime
    int        QCraneTime ;
    int        DueTime    ;
    int        ServiceTime;
    int        SourceDone ;        // 1 if the job has been pick up, otherwise 0
    int        DestDone   ;        // 1 if the job has been put down, otherwise 0
    String     SourceLocation;
    String     DestLocation  ;
    int        UseConstraintTime ;
};
struct  Vehicle
{
	int    Capacity;
        String StartLoc;
};
```
* It also generates __Trips, Tours,__ and __Destination information__.

## Miscellaneous Notes
- `DataSource` type is defined in `OpenPort` file
- `table2` is defined in `PortAGV` file.
- `Defines.h` contains all __constants__ (such as `MAXJOB_0 50, Maximum_Container_Jobs 40` etc.), and `Global_ext.h` contains __simulation variables__ (such as `SOURCEpOINT, dESTpOINT` etc.)
- `PSIMPLEX.h` Only consists of an interface `MCF_primal_net_simplex(MCF_network_p net)`. `PSIMPLEX1_3.cpp` implements it.
- BEA could stand for "Best Entering Arc".

## Borland Paradox DB files
- Each DataBase file is most likely created and compiled into a file with `.dfm` extension.
- To open DB files, download <a href="http://www.alexnolan.net/paradox/#google_vignette">this</a>
- There are various tables, but **PortLayoutTable** is the one generated with the <a href="#the-job-generator-hcdvrpcpp"> HCDVRP.cpp </a>
- `MCFAlgorithmForm->Table4` is `PortAGVTTable.DB` in the database.
- `PortAGVForm->Table2` and `PortAGVForm->Table1` are `portAGVTable.db` in the database. 
- `PortContainerForm->Table1` and `PortContainerForm->Table2` are `PortContainerTable.db` in the database.
- In BDE, `TTable` is the type for creating table objects.

## TODO List:
- [x] `mcfdefs.h`, and `mcf.h`
- [x] `PBLA1_3.cpp`, then `PSIMPLEX1_3.cpp`, and then `TREEUPS.cpp`
- [ ] `mcfutil.cpp`, `MCFLIGHT.cpp`, and `MCFModel`.
- [ ] The Job Generator
- [ ] `OUTPUT.cpp` as it is used by method `MCF_write_solution` in `MCFLIGHT.cpp`.
- [ ] `PortLayout`
- [ ] the functionalities of `MCFModel` 
- [ ] `PREPAIR`
- [ ] `PortBenchmark`, which is mostly a UI thing.
  
## Unanswered Questions:
- What is the use of `MCF_primal_iminus` (and hence `MCF_primal_net_simplex`)? what are jplus and iplus in them?
- Why are there 2 tables pointing to _the same DB_ in `PortAGV.cpp` and `PortContainer.cpp`?