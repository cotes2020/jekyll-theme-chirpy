--------------

### Javaisinsecure

---------------

### Java Insecure Deserialization

-----------------

- J2EE core relies on serialization such as->
 - Remote Method Invocation (RMI)
 - Java Management Extension
 - Java Message Service
- Java Server Faces (ViewState)
- Communication between JVMS
- Custom application protocol running on http

------------------

### Vulnerable Java Function

------------------

- `ObjectInputStream()`-:

```java
 try(ObjectInputStream ois =  new ObjectInputStream(new FileInputStream("user.ser"))) {
            ois.readObject();
            System.out.println("Object deserialized");
        } catch (Exception e) {
            e.printStackTrace();
        }
```

- 
