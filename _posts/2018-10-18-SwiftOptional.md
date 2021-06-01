---
layout: post
title: "Swift可选类型Optional的用法"
date: 2018-10-06 23:02:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Optional]
---

## Optional

```swift
let optional: String? = "unicorn"
var optional2: Int?
let optionals: [String?] = ["unicorn", nil, "dragon"]
var optionals2: [Int]?
```

## if let

```swift
let optional: String? = "unicorn"
if let value = optional {
  print(value)
}
```

## multiple variables

```swift
let optional: String? = "unicorn"
var optional2: Int?
if let value = optional, let value2 = optional2 {
  print(value)
  print(value2)
}
```

## boolean clause

```swift
let optional: String? = "unicorn"
if let value = optional, value.hasSuffix("corn") {
   print(value) // hasSuffix后缀结束
}
```

## guard

```swift
let optional: String? = "unicorn"
guard let value = optional else {    
	return
}
print(value)
```

## while let

```swift
let optional: String? = "unicorn"
while let value = optional {
  print(value) 
}
```

## nil coalescing

```swift
let optional: String? = "unicorn"
let value = optional ?? "nil"
print(value)
```

## force unwrapping

```swift
let optional: String? = "unicorn"
let value = optional!
print(value)
```

## switch block

```swift
let optional: String? = "unicorn"
switch optional { 
case .some(let value):   
	print(value) 
case .none:   
	print("nil") 
}
```

## map()

```swift
let value = optional.map(String.init(describing:))
```

## flatMap()

```swift
let value = optional.flatMap(URL.init(string:))
```

## compactMap()

```swift
let values = optionals.compactMap{ $0 }
```

## type casting

```swift	
let value = optional as! String
```

## optional chaining

```swift
let value = optional?.uppercased()
```

## for loop

```swift
for element in optionals {   
	if let value = element {}
}
```

## for case let

```swift
for case let optional? in optionals {}
```

## for case .some

```swift
for case .some(let value) in optionals {}
```

## forEach

```swift
optionals2?.forEach{ value in
}
```

## assignment

assigns if the optional has a value

```swift
optional2? = 2014
```

## pattern matching

```swift
switch optional { 
case "unicorn"?:   
	print("Unicorn!") 
default:   
	print("Not Unicorn")
}
```

## enums 

```swift
enum Animal {   
	case pet(type: String?) 
}
```

## switching with associated optionals

```swift
let enumValue = Animal.pet(type: optional)
switch enumValue {
case .pet(.some(let value)):
  print("I am a \(value).")
case .pet(.none):
  print("I am unknown.")
}
```

## switching on optional enums

```swift
let enumValue2: Animal? = nil
switch enumValue2 {
case .pet?:
  print("Pets")
default:
  print("No pets.")
}
```