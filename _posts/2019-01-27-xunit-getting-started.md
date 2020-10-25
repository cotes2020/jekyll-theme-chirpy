---
title: 'xUnit - Getting Started'
date: 2019-01-27T21:57:57+01:00
author: Wolfgang Ofner
categories: [Programming, DevOps]
tags: ['C#', TDD, xUnit]
---
In this post, I will explain the basics of xUnit and how to write unit tests with it. xUnit is an open source testing framework for the .Net framework and was written by the inventor of NUnit v2. More details can be found on <a href="https://xunit.github.io/" target="_blank" rel="noopener">xUnit&#8217;s Github page</a>. xUnit is used by .Net core as the default testing framework and its major advantage over NUnit is that every test runs in isolation, which makes it impossible that test influence each other.

## My Setup

For writing unit tests I use the following NuGet packages and extensions:

  * xUnit for unit testing
  * <a href="http://xbehave.github.io/" target="_blank" rel="noopener">xBehave</a> for acceptance tests (xBehave is based on xUnit)
  * <a href="https://fluentassertions.com/" target="_blank" rel="noopener">FluentAssertions</a> for more readable assertions
  * <a href="https://fakeiteasy.github.io/" target="_blank" rel="noopener">FakeItEasy</a> to create fake objects
  * <a href="https://resharper-plugins.jetbrains.com/packages/xunitcontrib/" target="_blank" rel="noopener">xUnit Resharper Extension</a> for xUnit shortcuts in Visual Studio

The code for today&#8217;s demo can be found on <a href="https://github.com/WolfgangOfner/xUnit-Getting-Started" target="_blank" rel="noopener">Github</a>. Keep in mind that the tests are only for the demonstration of xUnit. The tests are barely useful.

## Execute a test with xUnit

For each class I want to test, I create a separate class to which I add tests in the name, for example, if I want to test the Employee class I name my test class EmployeeTests. To create a test method, you only have to add the Fact attribute to the method.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Using-the-Fact-attritbue.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/01/Using-the-Fact-attritbue.jpg" alt="Using the xUnit Fact attribute" /></a>
  
  <p>
    Using the Fact attribute
  </p>
</div>

That&#8217;s all. You can run the test and if the constructor of your Employee class sets the salary to 1000, the test will pass. I like to name the object I want to test testee. Another common name is sut which stands for system under test.

## Reducing code duplication

In the intro, I mentioned that every test runs in isolation in xUnit. This is done by creating a new instance for each test. Therefore the constructor is called for each test and can be used to initialize objects, which are needed for the tests. Since I will need the object of the Employee class in all my tests, I can initialize it in the constructor and don&#8217;t have to write the same code over and over in every test.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Initialize-the-testee-in-the-constructor.jpg"><img aria-describedby="caption-attachment-1532" loading="lazy" class="size-full wp-image-1532" src="/assets/img/posts/2019/01/Initialize-the-testee-in-the-constructor.jpg" alt="Initialize the testee in the constructor" /></a>
  
  <p>
    Initialize the testee in the constructor
  </p>
</div>

## Cleaning up after tests

Sometimes you have to do some cleanup like a database rollback or deleting a file after the tests were executed. Like the constructor, this can be done in a central place for all tests. To do that implement the IDisposable interface and implement the Dispose method. This method is called every time a test is finished.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Implement-the-IDisposable-interface.jpg"><img aria-describedby="caption-attachment-1533" loading="lazy" class="size-full wp-image-1533" src="/assets/img/posts/2019/01/Implement-the-IDisposable-interface.jpg" alt="Implement the IDisposable interface" /></a>
  
  <p>
    Implement the IDisposable interface
  </p>
</div>

## Executing tests several times with different parameters

Often you want to execute a test with different parameters, for example, if a valid age for your employee has to be between at least 18 and maximum 65 years, you want to test the edge cases (17, 18, 65, 66). Additionally, you might test negative numbers. You could write several asserts but this would be a lot of typing and not really practical. The solution for this is the Theory attribute in xUnit. A Theory allows you to pass values from different sources as parameters to your test method. With the InlineData attribute, you can add values for the parameter.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Executing-the-same-method-with-several-input-variables.jpg"><img aria-describedby="caption-attachment-1534" loading="lazy" class="size-full wp-image-1534" src="/assets/img/posts/2019/01/Executing-the-same-method-with-several-input-variables.jpg" alt="Executing the same method with several input variables" /></a>
  
  <p>
    Executing the same method with several input variables
  </p>
</div>

If you run this test method, five test cases will be executed.

## Skipping a test

Sometimes you don&#8217;t want a test to be executed. To ignore tests, add the Skip attribute and provide an info message.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Skipping-a-test.jpg"><img aria-describedby="caption-attachment-1535" loading="lazy" class="size-full wp-image-1535" src="/assets/img/posts/2019/01/Skipping-a-test.jpg" alt="Skipping a test" /></a>
  
  <p>
    Skipping a test
  </p>
</div>

## Grouping tests together

I barely use this feature but sometimes you want to group certain tests together. This can be for example all tests from one class and only some tests from another class. To do that use the Trait attribute and provide a name and category. You can apply this attribute to a class or to a single test.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-method.jpg"><img aria-describedby="caption-attachment-1536" loading="lazy" class="size-full wp-image-1536" src="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-method.jpg" alt="Applying the Trait attribute to a test method" /></a>
  
  <p>
    Applying the Trait attribute to a test method
  </p>
</div>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-class.jpg"><img aria-describedby="caption-attachment-1537" loading="lazy" class="size-full wp-image-1537" src="/assets/img/posts/2019/01/Applying-the-Trait-attribute-to-a-test-class.jpg" alt="Applying the Trait attribute to a test class" /></a>
  
  <p>
    Applying the Trait attribute to a test class
  </p>
</div>

If you run the tests and group the output by category, all traits with the same category will be grouped together.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Grouped-test-output.jpg"><img aria-describedby="caption-attachment-1538" loading="lazy" class="size-full wp-image-1538" src="/assets/img/posts/2019/01/Grouped-test-output.jpg" alt="Grouped test output" /></a>
  
  <p>
    Grouped test output
  </p>
</div>

## Add information to the test result output

By default, no output is generated when a test finished. For reporting reasons, it can be useful to add some information on what the test did to the output of the test. This can be done with the ITestOutputHelper. Pass it as parameter in the constructor of your test class and initialize a private field with it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Using-the-ITestOutputHelper.jpg"><img aria-describedby="caption-attachment-1539" loading="lazy" class="size-full wp-image-1539" src="/assets/img/posts/2019/01/Using-the-ITestOutputHelper.jpg" alt="Using the ITestOutputHelper" /></a>
  
  <p>
    Using the ITestOutputHelper
  </p>
</div>

Next, use the WriteLine method of the ITestOutputHelper object to create the desired output.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Creating-a-custom-message-for-the-test-result.jpg"><img aria-describedby="caption-attachment-1540" loading="lazy" class="size-full wp-image-1540" src="/assets/img/posts/2019/01/Creating-a-custom-message-for-the-test-result.jpg" alt="Creating a custom message for the test result" /></a>
  
  <p>
    Creating a custom message for the test result
  </p>
</div>

When you run the test, you will see the message in the test result window.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Output-message-of-the-test.jpg"><img aria-describedby="caption-attachment-1541" loading="lazy" class="size-full wp-image-1541" src="/assets/img/posts/2019/01/Output-message-of-the-test.jpg" alt="Output message of the test" /></a>
  
  <p>
    Output message of the test
  </p>
</div>

## Share resources over multiple tests

Previously, I mentioned that for every test a new object is instantiated and therefore isolated from the other tests.  Sometimes you need to share a resource with several tests. This can be done with Fixtures. First, you have to create a so-called fixture class with the information you want to share. In my simple example, I set DateTime.Now to demonstrate that every test uses the same instance of the object.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Fixture-class-to-share-the-time-property.jpg"><img aria-describedby="caption-attachment-1542" loading="lazy" class="size-full wp-image-1542" src="/assets/img/posts/2019/01/Fixture-class-to-share-the-time-property.jpg" alt="Fixture class to share the time property" /></a>
  
  <p>
    Fixture class to share the time property
  </p>
</div>

Next, I am creating a collection class with the CollectionDefiniton attribute and the ICollectionFixture interface with my previously created fixture class.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Creating-the-collection-to-share-date-accross-tests.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/01/Creating-the-collection-to-share-date-accross-tests.jpg" alt="Creating the collection to share date across tests" /></a>
  
  <p>
    Creating the collection to share date across tests
  </p>
</div>

Finally, I add the Collection attribute with the previously set name to my test class and pass the fixture class in the constructor.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Passing-the-collection-to-the-test-class.jpg"><img aria-describedby="caption-attachment-1544" loading="lazy" class="size-full wp-image-1544" src="/assets/img/posts/2019/01/Passing-the-collection-to-the-test-class.jpg" alt="Passing the collection to the test class" /></a>
  
  <p>
    Passing the collection to the test class
  </p>
</div>

To demonstrate that the _timeFixture object stays the same, I run a couple of tests with Thread.Sleep(1500) and both tests will output the same time.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Tests-using-the-fixture-object.jpg"><img aria-describedby="caption-attachment-1546" loading="lazy" class="size-full wp-image-1546" src="/assets/img/posts/2019/01/Tests-using-the-fixture-object.jpg" alt="Tests using the fixture object" /></a>
  
  <p>
    Tests using the fixture object
  </p>
</div>

Both tests will print the same output.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/The-same-output-of-the-fixture-object.jpg"><img aria-describedby="caption-attachment-1545" loading="lazy" class="size-full wp-image-1545" src="/assets/img/posts/2019/01/The-same-output-of-the-fixture-object.jpg" alt="The same output of the fixture object" /></a>
  
  <p>
    The same output of the fixture object
  </p>
</div>

## Provide test data from a class

Previously, I showed how to use the Theory attribute to pass several parameters for the test. If you want the same data for several tests, you would have to enter it several times. This is error-prone and unpractical. Therefore, you can place these values in a class and just add a reference to the class.

Create a new class with a static property and only a getter which yield returns all your test data as object arrays.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Class-with-test-data.jpg"><img aria-describedby="caption-attachment-1547" loading="lazy" class="size-full wp-image-1547" src="/assets/img/posts/2019/01/Class-with-test-data.jpg" alt="Class with test data" /></a>
  
  <p>
    Class with test data
  </p>
</div>

For your test, use the MemberData instead of the InlineData attribute and provide the name of the property and the type of the class containing your test data.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Using-test-data-from-a-class.jpg"><img aria-describedby="caption-attachment-1548" loading="lazy" class="size-full wp-image-1548" src="/assets/img/posts/2019/01/Using-test-data-from-a-class.jpg" alt="Using test data from a class" /></a>
  
  <p>
    Using test data from a class
  </p>
</div>

## Provide test data with a custom attribute

A custom attribute works the same way as the MemberData attribute but it is even less to write in your test. Create a new class and inherit from the DataAttribute class. Then override the GetData method.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Override-the-GetData-method-from-the-DataAttribute-class.jpg"><img aria-describedby="caption-attachment-1549" loading="lazy" class="size-full wp-image-1549" src="/assets/img/posts/2019/01/Override-the-GetData-method-from-the-DataAttribute-class.jpg" alt="Override the GetData method from the DataAttribute class" /></a>
  
  <p>
    Override the GetData method from the DataAttribute class
  </p>
</div>

After you created the class, add the name of the class (without Attribute) as the attribute to your Theory. xUnit will recognize your attribute and call the GetData method.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Use-the-custom-attribute.jpg"><img aria-describedby="caption-attachment-1550" loading="lazy" class="size-full wp-image-1550" src="/assets/img/posts/2019/01/Use-the-custom-attribute.jpg" alt="Use the custom attribute" /></a>
  
  <p>
    Use the custom attribute
  </p>
</div>

## Provide test data from an external source

The last method to provide data for your tests is from an external source. To read the data from a csv file, I placed the csv file in the root folder of my project and created a class with a static property. In the getter of the property, I read the file, split the values and return them as object arrays. Don&#8217;t forget to set the Copy to Output Directory property of the csv file to Copy always or Copy if newer. Otherwise, the file won&#8217;t be copied when you compile your code and therefore won&#8217;t be found at runtime.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Read-data-from-a-csv-file.jpg"><img aria-describedby="caption-attachment-1551" loading="lazy" class="size-full wp-image-1551" src="/assets/img/posts/2019/01/Read-data-from-a-csv-file.jpg" alt="Read data from a csv file" /></a>
  
  <p>
    Read data from a csv file
  </p>
</div>

Now use the MemberData attribute for your test to add the name of the property and the type of your class.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/01/Provide-data-from-an-external-file-to-your-test.jpg"><img loading="lazy" src="/assets/img/posts/2019/01/Provide-data-from-an-external-file-to-your-test.jpg" alt="Provide data from an external file to your test" /></a>
  
  <p>
    Provide data from an external file to your test
  </p>
</div>

## Conclusion

In this post, I gave a quick overview of xUnit and explained how to get data from several sources and how to reduce duplicate code. For more information on xUnit, I can recommend the Pluralsight course &#8220;<a href="https://app.pluralsight.com/library/courses/dotnet-core-testing-code-xunit-dotnet-getting-started/table-of-contents" target="_blank" rel="noopener">Testing .NET Core Code with xUnit.net: Getting Started</a>&#8221; from Jason Robert.

You can find the code of my demo on <a href="https://github.com/WolfgangOfner/xUnit-Getting-Started" target="_blank" rel="noreferrer noopener" aria-label="You can find all my solutions on GitHub. (opens in a new tab)">GitHub</a>.