---
title: "C# Switch"
description: ""
categories: [컴퓨터, 프로그래밍]
tags: [CSharp]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-04-19. 00:46 # Init
# last_modified_at: 2025-04-19. 00:46
---

## 머리말

---

## 구현

---

### Switch Statement

```cs
int number = 2;
switch (number)
{
    case 1:
        Console.WriteLine("Number is 1");
        break;
    case 2:
        Console.WriteLine("Number is 2");
        break;
    default:
        Console.WriteLine("Number is not 1 or 2");
        break;
}
```

### Switch PatternMatching

```cs
object obj = 1;

switch (obj)
{
    case int i when i > 0 or i < 10:
        Console.WriteLine($"Integer: {i}");
        break;
    case string s:
        Console.WriteLine($"String: {s}");
        break;
    default:
        Console.WriteLine("Unknown type");
        break;
}
```

### Switch Expression

```cs
int number = 5;
int result = number switch
{
    1 => 10,
    2 => 20,
    _ when number > 0 => 30,
    _ => 0
};
```

### 활용

```cs
int x = 1;
int y = 2;
int z = 3;
int result = (x, y) switch
{
    (1, 2) => 10,
    (2, 3) => 20,
    _ => 0
};
```

```cs
var obj = new { Name = "Kururu", Age = 20 };
switch (obj)
{
    case { Name: "Kururu", Age: 20 }:
        Console.WriteLine("Matched Kururu, 20");
        break;
    case { Name: "Kururu" }:
        Console.WriteLine("Matched Kururu");
        break;
    default:
        Console.WriteLine("No match");
        break;
}

var result = obj switch
{
    { Name: "Kururu", Age: 20 } => "Matched Kururu, 20",
    { Name: "Kururu", Age: > 20 } => "Matched Kururu, Age > 20",
    { Name: "Kururu" } => "Matched Kururu",
    _ => "No match",
};
```
