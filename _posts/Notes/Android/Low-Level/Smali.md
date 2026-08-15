--------------

### Smali

--------------

- Some methods don't want precise values, they have to instances of the Java classese.g

```smali
#int
const/16 v0, 0x7fff

invoke-static {v0}, Ljava/lang/Integer;->valueOf(I)Ljava/lang/Integer;
move-result-object p1
```
```smali
# 1. Load the raw boolean true (1) into v0
    const/4 v0, 0x1

    # 2. Convert that primitive 1 into a Boolean OBJECT
    invoke-static {v0}, Ljava/lang/Boolean;->valueOf(Z)Ljava/lang/Boolean;
    move-result-object p1
```
--------------
