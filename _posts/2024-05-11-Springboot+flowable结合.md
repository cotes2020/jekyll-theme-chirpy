---
title: Spring boot flowable Demo
categories: [Java]
tags: [Spring Boot,Flowable]
date: 2024-05-11
---


## Springboot+flowable结合 

Demo 环境为Java 17+springboot2.7.17+flowable6.8.0  
代码仓库 https://github.com/wsleepybear/flowableDemo

### 1. 前置准备

#### 1.1 flowable表生成

在Maven项目的pom.xml文件中添加Spring Boot和Flowable的依赖。

```xml
<dependency>  
    <groupId>org.springframework.boot</groupId>  
    <artifactId>spring-boot-starter</artifactId>  
</dependency>
<dependency>
    <groupId>org.flowable</groupId>
    <artifactId>flowable-spring-boot-starter</artifactId>
    <version>6.8.0</version>
</dependency>
<!--        或者-->
<dependency>
     <groupId>org.flowable</groupId>
     <artifactId>flowable-spring-boot-starter-rest</artifactId>
     <version>6.8.0</version>
</dependency>
```

创建一个Java类，定义一个Main方法，用于初始化Flowable的ProcessEngine。这段代码只需要执行一次，会连接到指定的数据库，如果库中没有Flowable对应数据库，或结构发生变化，会自动创建或更新表。

```java
public class InitFlowable {
    private static final ProcessEngine processEngine;
    static {
        // 单机处理引擎配置信息实例
        StandaloneProcessEngineConfiguration cfg = new StandaloneProcessEngineConfiguration();
        // 连接数据库的信息
        cfg.setJdbcUrl("jdbc:mysql://*****:3306/flowable?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC");
        cfg.setJdbcUsername("******");
        cfg.setJdbcPassword("******");
        cfg.setJdbcDriver("com.mysql.cj.jdbc.Driver");
        // 设置了true，确保在JDBC参数连接的数据库中，数据库表结构不存在时，会创建相应的表结构。
        cfg.setDatabaseSchemaUpdate(ProcessEngineConfiguration.DB_SCHEMA_UPDATE_TRUE);
        // 通过配置获取执行引擎
        processEngine = cfg.buildProcessEngine();
    }

    public static void main(String[] args) {
        System.out.println(processEngine);
    }

}
```


##### 1.1.1 生成的数据表

Flowable的所有数据库表都以**ACT_** 开头。第二部分是说明表用途的两字符标示符。服务API的命名也大略符合这个规则。

ACT_RE_: 'RE’代表repository。带有这个前缀的表包含“静态”信息，例如流程定义与流程资源（图片、规则等）。
ACT_RU_: 'RU’代表runtime。这些表存储运行时信息，例如流程实例（process instance）、用户任务（user task）、变量（variable）、作业（job）等。Flowable只在流程实例运行中保存运行时数据，并在流程实例结束时删除记录。这样保证运行时表小和快。
ACT_HI_: 'HI’代表history。这些表存储历史数据，例如已完成的流程实例、变量、任务等。
ACT_GE_: 通用数据。在多处使用。

#### 1.2 定义实体类
定义实体类，包含了流程创建过程中的相关信息，这个类可以作为Controller的参数，接收用户在前端定义的节点信息。
```java
@Data
public class CreateProcessDefinitionLineDTO implements Serializable {
    private static final long serialVersionUID = 8721595101258489619L;

    @ApiModelProperty(value = "流程定义名称")
    private String flowName;

    @ApiModelProperty(value = "流程类型",notes = "请假,报销等")
    private String category;

    @ApiModelProperty(value = "流程节点定义")
    private List<ProcessNodeDefinitionDTO> processNodeDefinitionDTOS;

}

@Data
public class ProcessNodeDefinitionDTO implements Serializable {
    private static final long serialVersionUID = -1916624505173789546L;

    @ApiModelProperty(value = "节点定义名称")
    private String nodeName;

    @ApiModelProperty(value = "节点类型")
    private String nodeType;

}
```

### 2. BPMN创建到流程结束的Baseline
在本demo中，我们使用员工请假的例子来实例，请假流程为：  
员工申请->hr批准->主管批准->流程结束。  
flowable的流程为:
1. 流程创建
2. 流程实例启动
3. 任务执行
4. 流程实例结束

#### 2.1 流程创建
##### 2.1.1 创建一个流程并且设置开始节点
定义流程，并设置开始节点，在开始节点之后可以设置一个初始服务任务。
```java
    Process process = new Process();

    process.setId(createProcessDefinitionLineDTO.getFlowName());
    process.setName(createProcessDefinitionLineDTO.getFlowName());
    // 开始节点
    StartEvent startEvent = new StartEvent();
    startEvent.setId("StartEvent"+ UUID.randomUUID().toString().substring(0, 11));
    startEvent.setName("开始");
    process.addFlowElement(startEvent);
    // 将上一个节点的暂存下来，用于后续执行顺序的定义
    FlowElement previousElement = startEvent;

    ServiceTask startServiceTask = new ServiceTask();
    startServiceTask.setId("StartServiceTask"+UUID.randomUUID().toString().substring(0,11));
    startServiceTask.setName("提交");
    startServiceTask.setImplementationType(ImplementationType.IMPLEMENTATION_TYPE_CLASS);
    startServiceTask.setImplementation(StartServiceTask.class.getName());
    process.addFlowElement(startServiceTask);
```
##### 2.1.2 定义用户任务
在开始节点节点后，既可以添加服务任务，也可以添加用户任务，在基线Demo中我们不搭建服务任务。
```java
    for (ProcessNodeDefinitionDTO form : createProcessDefinitionLineDTO.getProcessNodeDefinitionDTOS()) {
        if ("userTask".equals(form.getNodeType())){
            UserTask userTask = new UserTask();
            userTask.setId("UserTask"+ UUID.randomUUID().toString().substring(0, 11));
            userTask.setName(form.getNodeName());
            userTask.setAssignee("${" + form.getNodeName() + "Assignee}");
            process.addFlowElement(userTask);


        }
    }
```

将节点与节点连接在一起，以开始服务任务节点与用户任务节点连接为例。
```java
    SequenceFlow sequenceFlow = new SequenceFlow();
    sequenceFlow.setSourceRef(previousElement.getId());
    sequenceFlow.setTargetRef(userTask.getId());
    process.addFlowElement(sequenceFlow);
    previousElement = userTask;
``` 

将结束节点与前面的节点连接起来。
```java
    EndEvent endEvent = new EndEvent();
    endEvent.setId("endEvent"+UUID.randomUUID().toString().substring(0,11));
    endEvent.setName("结束");
    process.addFlowElement(endEvent);

    SequenceFlow sequenceFlowToEnd = new SequenceFlow();
    sequenceFlowToEnd.setSourceRef(previousElement.getId());
    sequenceFlowToEnd.setTargetRef(endEvent.getId());
    process.addFlowElement(sequenceFlowToEnd);
```


整个流程的目的是将一个新的流程定义添加到Flowable流程引擎中。下面这代码主要是在创建和部署一个BPMN模型到Flowable流程引擎中。首先，创建一个BpmnModel对象并向其中添加一个流程定义。然后，使用BpmnAutoLayout对BPMN模型进行自动布局优化。接着，创建一个部署对象，将BPMN模型以XML格式添加到部署对象中，并设置部署的名称和类别。然后，将部署对象部署到Flowable的流程引擎中，并获取新部署的流程定义。最后，设置流程定义的类别。
```java
    BpmnModel bpmnModel = new BpmnModel();
    bpmnModel.addProcess(process);
    if (bpmnModel == null) {
        throw new IllegalArgumentException("bpmnModel cannot be null");
    }
    new BpmnAutoLayout(bpmnModel).execute();

    Deployment deployment = repositoryService.createDeployment()
            .addBpmnModel(createProcessDefinitionLineDTO.getFlowName() + ".bpmn20.xml", bpmnModel)
            .name(createProcessDefinitionLineDTO.getFlowName())
            .category(createProcessDefinitionLineDTO.getCategory())
            .deploy();
    ProcessDefinition processDefinition = repositoryService.createProcessDefinitionQuery()
            .deploymentId(deployment.getId())
            .singleResult();

    repositoryService.setProcessDefinitionCategory(
            processDefinition.getId(), createProcessDefinitionLineDTO.getCategory());
```

[完整代码，从此跳转](https://github.com/wsleepybear/flowableDemo/blob/master/src/main/java/com/example/flowabledemo/service/impl/FlowableProcessBaselineServiceImpl.java)

